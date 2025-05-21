import torch
import re
import math # Make sure math is imported for isclose
from blackhole.embedding import decode_number_from_features # Import your embedding decoding function
# We don't import tokenize here, as it's used for input preparation, not prediction output processing.

def predict_and_decode_answer(model, encoder_token_ids, encoder_numeric_features,
                              encoder_attention_mask, vocab, device, max_decoding_len=128, padded_feat_row=None):
    """
    Generates and decodes answers from the model using greedy decoding.

    Args:
        model (torch.nn.Module): The trained model.
        encoder_token_ids (torch.Tensor): Input token IDs for the encoder (Batch_size, Seq_len).
        encoder_numeric_features (torch.Tensor): Input numerical features for the encoder (Batch_size, Seq_len, Feature_dim).
        encoder_attention_mask (torch.Tensor): Attention mask for the encoder (Batch_size, Seq_len).
        vocab (dict): The vocabulary mapping tokens to IDs.
        device (torch.device): The device (CPU/GPU) for computations.
        max_decoding_len (int): Maximum length of the generated sequence.
        padded_feat_row (torch.Tensor, optional): A tensor representing padded numerical features.
                                                   If None, it will be initialized.

    Returns:
        list: A list of decoded answer strings for each example in the batch.
    """
    model.eval() # Set model to evaluation mode

    idx_to_token = {idx: token for token, idx in vocab.items()}
    num_token_id = vocab.get('<|num|>', vocab.get('<|unk|>', 0))
    bos_token_id = vocab.get('<|bos|>', vocab.get('<|unk|>', 0))
    eos_token_id = vocab.get('<|eos|>', vocab.get('<|unk|>', 0))

    # Initialize padded_feat_row if not provided.
    # This must match the model's expected feature_dim.
    if padded_feat_row is None:
        if not hasattr(model, 'feature_dim'):
            raise AttributeError("model must have 'feature_dim' attribute or 'padded_feat_row' must be provided.")
        padded_feat_row = torch.full((model.feature_dim,), -2.0, dtype=torch.float32).to(device)

    batch_size = encoder_token_ids.size(0)

    # Initialize decoder inputs
    decoder_input_token_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    decoder_input_numeric_features = padded_feat_row.unsqueeze(0).repeat(batch_size, 1, 1) # [B, 1, feature_dim]
    decoder_attention_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device) # No padding initially

    generated_tokens_list = [[] for _ in range(batch_size)]
    generated_num_values_list = [[] for _ in range(batch_size)]

    with torch.no_grad(): # Disable gradient calculations for inference
        for _ in range(max_decoding_len):
            # Forward pass through the model
            token_logits, num_feature_output = model(
                encoder_token_ids=encoder_token_ids,
                encoder_numeric_features=encoder_numeric_features,
                encoder_attention_mask=encoder_attention_mask,
                decoder_token_ids=decoder_input_token_ids,
                decoder_numeric_features_input=decoder_input_numeric_features,
                decoder_attention_mask=decoder_attention_mask
            )

            # Get the last predicted token and numeric feature for the current step
            next_token_logits = token_logits[:, -1, :] # [B, vocab_size]
            next_num_features = num_feature_output[:, -1, :] # [B, feature_dim]

            # Get the token with the highest probability (greedy decoding)
            next_token_ids = torch.argmax(next_token_logits, dim=-1) # [B]

            # Append the newly predicted token to the decoder input for the next step
            decoder_input_token_ids = torch.cat([decoder_input_token_ids, next_token_ids.unsqueeze(1)], dim=1)

            # Prepare numeric features for the next step's input
            decoder_input_numeric_features_next_step = torch.empty(batch_size, 1, model.feature_dim, device=device)
            for b_idx in range(batch_size):
                if next_token_ids[b_idx].item() == num_token_id:
                    # If the predicted token is <|num|>, use its predicted features
                    decoder_input_numeric_features_next_step[b_idx, 0, :] = next_num_features[b_idx, :]
                else:
                    # Otherwise, use the padded feature row
                    decoder_input_numeric_features_next_step[b_idx, 0, :] = padded_feat_row

            decoder_input_numeric_features = torch.cat([decoder_input_numeric_features, decoder_input_numeric_features_next_step], dim=1)
            # Update attention mask (no additional padding needed for this decoding phase)
            decoder_attention_mask = torch.cat([decoder_attention_mask, torch.zeros((batch_size, 1), dtype=torch.bool, device=device)], dim=1)


            # Process generated tokens for each example in the batch
            for i in range(batch_size):
                # Check for EOS token to stop decoding for a particular example
                if next_token_ids[i].item() == eos_token_id:
                    if not generated_tokens_list[i] or generated_tokens_list[i][-1] != 'STOP_DECODING':
                        generated_tokens_list[i].append('STOP_DECODING') # Custom signal to stop processing this sequence
                elif generated_tokens_list[i] and generated_tokens_list[i][-1] == 'STOP_DECODING':
                    continue # This sequence has already stopped decoding
                else:
                    # Convert token ID to token string
                    token = idx_to_token.get(next_token_ids[i].item(), '<|unk|>')
                    generated_tokens_list[i].append(token)
                    if token == '<|num|>':
                        # Decode the numerical value from its features
                        decoded_val = decode_number_from_features(next_num_features[i].cpu().numpy())
                        generated_num_values_list[i].append(decoded_val)
                    else:
                        generated_num_values_list[i].append(None) # Placeholder for non-numerical tokens

            # Break if all sequences in the batch have generated EOS
            if all('STOP_DECODING' in seq for seq in generated_tokens_list):
                break

    # Post-processing: Reconstruct the answer string from generated tokens and numbers
    final_decoded_answers = []
    for i in range(batch_size):
        predicted_answer_tokens = []
        current_generated_num_idx = 0
        # Filter out the 'STOP_DECODING' signal before post-processing
        temp_generated_tokens = [tok for tok in generated_tokens_list[i] if tok != 'STOP_DECODING']

        k = 0
        while k < len(temp_generated_tokens):
            token = temp_generated_tokens[k]

            if token == '<|num|>':
                val = generated_num_values_list[i][current_generated_num_idx]
                if val is not None:
                    # Format numbers: integer if effectively an integer, else two decimal places
                    if abs(val - round(val)) < 1e-6:
                        predicted_answer_tokens.append(str(int(round(val))))
                    else:
                        predicted_answer_tokens.append(f"{val:.2f}")
                current_generated_num_idx += 1
            elif token == '<|cap|>':
                # Capitalize the next valid word token
                if k + 1 < len(temp_generated_tokens) and temp_generated_tokens[k+1] not in ['<|cap|>', '<|allcaps|>', '<|num|>', '<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|space|>']:
                    predicted_answer_tokens.append(temp_generated_tokens[k+1].capitalize())
                    k += 1 # Skip the next token as it's been processed
            elif token == '<|allcaps|>':
                # Uppercase the next valid word token
                if k + 1 < len(temp_generated_tokens) and temp_generated_tokens[k+1] not in ['<|cap|>', '<|allcaps|>', '<|num|>', '<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|space|>']:
                    predicted_answer_tokens.append(temp_generated_tokens[k+1].upper())
                    k += 1 # Skip the next token as it's been processed
            elif token == '<|space|>':
                predicted_answer_tokens.append(' ')
            elif token == '<|pad|>':
                pass # Ignore padding tokens
            else:
                predicted_answer_tokens.append(token)
            k += 1

        # Reconstruct the predicted answer string and clean up whitespace/punctuation
        predicted_answer_raw = "".join(predicted_answer_tokens)
        predicted_answer_cleaned = re.sub(r'\s([.,!?;:])', r'\1', predicted_answer_raw) # Remove space before punctuation
        predicted_answer_cleaned = re.sub(r'\s+', ' ', predicted_answer_cleaned).strip() # Consolidate multiple spaces and strip leading/trailing

        final_decoded_answers.append(predicted_answer_cleaned)

    return final_decoded_answers
import torch
import re
import numpy as np # For float('nan') checks

# Assuming blackhole.embedding is correctly installed and accessible
from blackhole.embedding import decode_number_from_features

# Import special tokens from config
from config import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN,
    CAP_TOKEN, ALLCAPS_TOKEN, SPACE_TOKEN
)

def predict_and_decode_answer(model, encoder_token_ids, encoder_numeric_features, encoder_attention_mask, vocab, device, max_decoding_len=128):
    """
    Generates an answer from the model given encoded input.
    Decodes predicted tokens and numerical features into a human-readable string.
    """
    model.eval()
    idx_to_token = {idx: token for token, idx in vocab.items()}
    num_token_id = vocab.get(NUM_TOKEN, vocab.get(UNK_TOKEN, 0))
    bos_token_id = vocab.get(BOS_TOKEN, vocab.get(UNK_TOKEN, 0))
    eos_token_id = vocab.get(EOS_TOKEN, vocab.get(UNK_TOKEN, 0))
    pad_token_id = vocab.get(PAD_TOKEN, 0)
    
    # Create padded_feat_row dynamically based on model's feature_dim
    if not hasattr(model, 'feature_dim'):
        raise ValueError("Model must have a 'feature_dim' attribute to determine padded_feat_row.")
    padded_feat_row = torch.full((model.feature_dim,), -2.0, dtype=torch.float32).to(device)

    batch_size = encoder_token_ids.size(0)

    decoder_input_token_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    decoder_input_numeric_features = padded_feat_row.unsqueeze(0).repeat(batch_size, 1, 1)
    decoder_attention_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device) # False means not masked

    generated_tokens_list = [[] for _ in range(batch_size)]
    generated_num_values_list = [[] for _ in range(batch_size)]

    with torch.no_grad():
        for _ in range(max_decoding_len):
            token_logits, num_feature_output = model(
                encoder_token_ids=encoder_token_ids,
                encoder_numeric_features=encoder_numeric_features,
                encoder_attention_mask=encoder_attention_mask,
                decoder_token_ids=decoder_input_token_ids,
                decoder_numeric_features_input=decoder_input_numeric_features,
                decoder_attention_mask=decoder_attention_mask
            )

            next_token_logits = token_logits[:, -1, :]
            next_num_features = num_feature_output[:, -1, :]

            next_token_ids = torch.argmax(next_token_logits, dim=-1)

            decoder_input_token_ids = torch.cat([decoder_input_token_ids, next_token_ids.unsqueeze(1)], dim=1)
            
            decoder_input_numeric_features_next_step = torch.empty(batch_size, 1, model.feature_dim, device=device)
            for b_idx in range(batch_size):
                if next_token_ids[b_idx].item() == num_token_id:
                    decoder_input_numeric_features_next_step[b_idx, 0, :] = next_num_features[b_idx, :]
                else:
                    decoder_input_numeric_features_next_step[b_idx, 0, :] = padded_feat_row
            
            decoder_input_numeric_features = torch.cat([decoder_input_numeric_features, decoder_input_numeric_features_next_step], dim=1)
            decoder_attention_mask = torch.cat([decoder_attention_mask, torch.zeros((batch_size, 1), dtype=torch.bool, device=device)], dim=1)

            # Check if all sequences have generated EOS
            all_done = True
            for i in range(batch_size):
                if next_token_ids[i].item() == eos_token_id:
                    if not generated_tokens_list[i] or generated_tokens_list[i][-1] != 'STOP_DECODING':
                        generated_tokens_list[i].append('STOP_DECODING')
                elif generated_tokens_list[i] and generated_tokens_list[i][-1] == 'STOP_DECODING':
                    # This sequence has already finished decoding
                    pass
                else:
                    all_done = False # At least one sequence is still decoding
                    token = idx_to_token.get(next_token_ids[i].item(), UNK_TOKEN)
                    generated_tokens_list[i].append(token)
                    if token == NUM_TOKEN:
                        decoded_val = decode_number_from_features(next_num_features[i].cpu().numpy())
                        generated_num_values_list[i].append(decoded_val)
                    else:
                        generated_num_values_list[i].append(None) # Store None for non-numeric tokens

            if all_done:
                break

    final_decoded_answers = []
    for i in range(batch_size):
        predicted_answer_tokens = []
        current_generated_num_idx = 0
        temp_generated_tokens = [tok for tok in generated_tokens_list[i] if tok != 'STOP_DECODING']

        k = 0
        while k < len(temp_generated_tokens):
            token = temp_generated_tokens[k]
            if token == NUM_TOKEN:
                if current_generated_num_idx < len(generated_num_values_list[i]):
                    val = generated_num_values_list[i][current_generated_num_idx]
                    if val is not None and not np.isnan(val): # Check for None and NaN
                        if abs(val - round(val)) < 1e-6:
                            predicted_answer_tokens.append(str(int(round(val))))
                        else:
                            predicted_answer_tokens.append(f"{val:.2f}") # Format floats to 2 decimal places
                    else:
                        predicted_answer_tokens.append("<INVALID_NUM>") # Placeholder for invalid numbers
                else:
                    predicted_answer_tokens.append("<MISSING_NUM_DATA>") # Should not happen if lists are aligned
                current_generated_num_idx += 1
            elif token == CAP_TOKEN:
                if k + 1 < len(temp_generated_tokens) and temp_generated_tokens[k+1] not in SPECIAL_TOKENS:
                    predicted_answer_tokens.append(temp_generated_tokens[k+1].capitalize())
                    k += 1 # Skip next token as it was handled
                else:
                    predicted_answer_tokens.append(token) # Keep special token if nothing to capitalize
            elif token == ALLCAPS_TOKEN:
                if k + 1 < len(temp_generated_tokens) and temp_generated_tokens[k+1] not in SPECIAL_TOKENS:
                    predicted_answer_tokens.append(temp_generated_tokens[k+1].upper())
                    k += 1 # Skip next token as it was handled
                else:
                    predicted_answer_tokens.append(token) # Keep special token if nothing to capitalize
            elif token == SPACE_TOKEN:
                predicted_answer_tokens.append(' ')
            elif token == PAD_TOKEN or token == BOS_TOKEN: # BOS should not appear in generated output directly
                pass
            else:
                predicted_answer_tokens.append(token)
            k += 1
        
        predicted_answer_raw = "".join(predicted_answer_tokens)
        # Clean up spaces around punctuation and multiple spaces
        predicted_answer_cleaned = re.sub(r'\s([.,!?;:])', r'\1', predicted_answer_raw)
        predicted_answer_cleaned = re.sub(r'\s+', ' ', predicted_answer_cleaned).strip()
        final_decoded_answers.append(predicted_answer_cleaned)

    return final_decoded_answers

def decode_token_ids_to_text(token_ids, vocab):
    """
    Helper function to decode a list of token IDs back into a string for display.
    This is different from the blackhole_tokenizer.tokenize used for data prep.
    """
    idx_to_token = {idx: token for token, idx in vocab.items()}
    # Filter out pad, bos, eos for cleaner display
    filtered_tokens = [
        idx_to_token.get(idx, UNK_TOKEN)
        for idx in token_ids
        if idx not in [vocab.get(PAD_TOKEN), vocab.get(BOS_TOKEN), vocab.get(EOS_TOKEN)]
    ]
    return ' '.join(filtered_tokens)
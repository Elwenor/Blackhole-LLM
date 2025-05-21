import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
from blackhole.embedding import decode_number_from_features # Import this for evaluate
from blackhole.tokenizer import tokenize # Import this for evaluate
from torch.utils.data import DataLoader # Explicitly import DataLoader


# --- LOSS FUNCTIONS ---
def focal_loss(logits, targets, alpha=0.25, gamma=2.0, ignore_index=0):
    """Calculates focal loss for classification."""
    ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()

def mse_loss_for_numerical_features(pred_features, target_features, num_mask):
    """Calculates MSE loss for numerical features, masked by num_mask."""
    num_mask_expanded = num_mask.unsqueeze(-1).float()

    # Apply mask to select only relevant predictions and targets
    masked_pred_features = pred_features * num_mask_expanded
    masked_target_features = target_features * num_mask_expanded

    diff = masked_pred_features - masked_target_features
    sq_diff = diff ** 2

    num_active_elements = num_mask.sum() * pred_features.size(-1)

    if num_active_elements > 0:
        return sq_diff.sum() / num_active_elements
    else:
        return torch.tensor(0.0, device=pred_features.device)


# --- TRAINING AND EVALUATION LOGIC ---

def train_step(model, optimizer, scheduler, batch, vocab, device, clip_grad=1.0):
    """Performs one training step."""
    model.train()

    # Move batch to device
    encoder_token_ids = batch['encoder_token_ids'].to(device)
    encoder_numeric_features = batch['encoder_numeric_features'].to(device)
    encoder_attention_mask = batch['encoder_attention_mask'].to(device) # This is already boolean (True for pad)
    decoder_input_token_ids = batch['decoder_input_token_ids'].to(device)
    decoder_input_numeric_features = batch['decoder_input_numeric_features'].to(device)
    decoder_output_token_targets = batch['decoder_output_token_targets'].to(device)
    decoder_output_numeric_targets = batch['decoder_output_numeric_targets'].to(device)
    decoder_attention_mask = batch['decoder_attention_mask'].to(device) # This is already boolean (True for pad)

    optimizer.zero_grad()

    token_logits, num_feature_output = model(
        encoder_token_ids=encoder_token_ids,
        encoder_numeric_features=encoder_numeric_features,
        encoder_attention_mask=encoder_attention_mask,
        decoder_token_ids=decoder_input_token_ids,
        decoder_numeric_features_input=decoder_input_numeric_features,
        decoder_attention_mask=decoder_attention_mask
    )

    pad_id = vocab.get('<|pad|>', 0)
    num_id = vocab.get('<|num|>', vocab.get('<|unk|>', 0)) # Fallback for num_id if not found

    ce_loss = focal_loss(token_logits.view(-1, token_logits.size(-1)),
                         decoder_output_token_targets.view(-1),
                         ignore_index=pad_id)

    num_target_mask = (decoder_output_token_targets == num_id)
    numeric_loss = mse_loss_for_numerical_features(
        num_feature_output,
        decoder_output_numeric_targets,
        num_target_mask
    )

    # Total loss: Adjust weights as needed
    loss = ce_loss + 0.5 * numeric_loss # Increased weight for numeric loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {
        'ce_loss': ce_loss.item(),
        'numeric_loss': numeric_loss.item(),
        'total_loss': loss.item()
    }


def evaluate(model, dataset, vocab, device, batch_size=32, max_decoding_len=128, collate_fn=None):
    """
    Evaluates the model on a given dataset (validation/test).
    Performs greedy decoding and calculates exact match and numerical accuracy.
    """
    model.eval()
    idx_to_token = {idx: token for token, idx in vocab.items()}
    num_token_id = vocab.get('<|num|>', vocab.get('<|unk|>', 0))
    bos_token_id = vocab.get('<|bos|>', vocab.get('<|unk|>', 0))
    eos_token_id = vocab.get('<|eos|>', vocab.get('<|unk|>', 0))
    pad_token_id = vocab.get('<|pad|>', 0)
    # Placeholder for numeric features when generating
    padded_feat_row = torch.full((model.feature_dim,), -2.0, dtype=torch.float32).to(device)

    correct_exact_matches = 0
    correct_numerical_matches = 0
    total_examples = 0

    # Use the passed collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn # Use the passed collate_fn here
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            encoder_token_ids = batch['encoder_token_ids'].to(device)
            encoder_numeric_features = batch['encoder_numeric_features'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)

            batch_size = encoder_token_ids.size(0)

            # Initialize decoder input with <|bos|>
            decoder_input_token_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
            decoder_input_numeric_features = padded_feat_row.unsqueeze(0).repeat(batch_size, 1, 1) # [B, 1, feature_dim]
            decoder_attention_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device) # No padding initially

            predicted_token_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            predicted_numeric_features = torch.empty((batch_size, 0, model.feature_dim), dtype=torch.float32, device=device)
            generated_sequences = [[] for _ in range(batch_size)] # Store tokens for reconstruction
            generated_num_values = [[] for _ in range(batch_size)] # Store decoded numbers

            for _ in range(max_decoding_len):
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

                next_token_ids = torch.argmax(next_token_logits, dim=-1) # [B]

                # Append to predicted sequences
                predicted_token_ids = torch.cat([predicted_token_ids, next_token_ids.unsqueeze(1)], dim=1)
                predicted_numeric_features = torch.cat([predicted_numeric_features, next_num_features.unsqueeze(1)], dim=1)

                # Prepare for next step: extend decoder input
                decoder_input_token_ids = torch.cat([decoder_input_token_ids, next_token_ids.unsqueeze(1)], dim=1)
                # For the next step's input, if the current token is <|num|>, use its predicted features
                # Otherwise, use the padded_feat_row.
                decoder_input_numeric_features_next_step = torch.empty(batch_size, 1, model.feature_dim, device=device)
                for b_idx in range(batch_size):
                    if next_token_ids[b_idx].item() == num_token_id:
                        decoder_input_numeric_features_next_step[b_idx, 0, :] = next_num_features[b_idx, :]
                    else:
                        decoder_input_numeric_features_next_step[b_idx, 0, :] = padded_feat_row
                
                decoder_input_numeric_features = torch.cat([decoder_input_numeric_features, decoder_input_numeric_features_next_step], dim=1)
                decoder_attention_mask = torch.cat([decoder_attention_mask, torch.zeros((batch_size, 1), dtype=torch.bool, device=device)], dim=1)


                # Process generated tokens for each example in the batch
                for i in range(batch_size):
                    if next_token_ids[i].item() == eos_token_id:
                        # Once EOS is generated, stop decoding for this example
                        # and mark it as complete
                        if len(generated_sequences[i]) == 0 or generated_sequences[i][-1] != 'STOP_DECODING':
                            generated_sequences[i].append('STOP_DECODING') # Custom stop signal
                    elif generated_sequences[i] and generated_sequences[i][-1] == 'STOP_DECODING':
                        continue # Already stopped for this example
                    else:
                        token = idx_to_token.get(next_token_ids[i].item(), '<|unk|>')
                        generated_sequences[i].append(token)
                        if token == '<|num|>':
                            # Decode the number from its predicted features
                            decoded_val = decode_number_from_features(next_num_features[i].cpu().numpy())
                            generated_num_values[i].append(decoded_val)
                        else:
                            generated_num_values[i].append(None) # Placeholder for non-numerical tokens

                # Check if all sequences have generated EOS
                if all('STOP_DECODING' in seq for seq in generated_sequences):
                    break # All sequences are done

            # Calculate accuracy for the batch
            for i in range(batch_size):
                total_examples += 1
                predicted_answer_tokens = []
                # Use a separate index for generated_num_values as it only stores numbers
                current_generated_num_idx = 0
                
                # Create a mutable list of tokens for capitalization handling
                temp_generated_tokens = [tok for tok in generated_sequences[i] if tok != 'STOP_DECODING']

                k = 0
                while k < len(temp_generated_tokens):
                    token = temp_generated_tokens[k]

                    if token == '<|num|>':
                        val = generated_num_values[i][current_generated_num_idx]
                        if val is not None:
                            if abs(val - round(val)) < 1e-6:
                                predicted_answer_tokens.append(str(int(round(val))))
                            else:
                                predicted_answer_tokens.append(f"{val:.2f}") # Format floats for consistency
                        current_generated_num_idx += 1
                    elif token == '<|cap|>':
                        # Look at the next token to capitalize it
                        if k + 1 < len(temp_generated_tokens) and temp_generated_tokens[k+1] not in ['<|cap|>', '<|allcaps|>', '<|num|>', '<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|space|>']:
                            predicted_answer_tokens.append(temp_generated_tokens[k+1].capitalize())
                            k += 1 # Skip the next token as it's been processed
                        # else: just drop <|cap|> if no word follows
                    elif token == '<|allcaps|>':
                        # Look at the next token to uppercase it
                        if k + 1 < len(temp_generated_tokens) and temp_generated_tokens[k+1] not in ['<|cap|>', '<|allcaps|>', '<|num|>', '<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|space|>']:
                            predicted_answer_tokens.append(temp_generated_tokens[k+1].upper())
                            k += 1 # Skip the next token as it's been processed
                        # else: just drop <|allcaps|> if no word follows
                    elif token == '<|space|>':
                        predicted_answer_tokens.append(' ')
                    elif token == '<|pad|>':
                        pass # Skip pad tokens
                    else:
                        predicted_answer_tokens.append(token)
                    k += 1


                # Reconstruct the predicted answer string
                predicted_answer_raw = "".join(predicted_answer_tokens)
                predicted_answer_cleaned = re.sub(r'\s([.,!?;:])', r'\1', predicted_answer_raw)
                predicted_answer_cleaned = re.sub(r'\s+', ' ', predicted_answer_cleaned).strip()


                # Get true answer value
                true_answer_string = batch['original_answer_strings'][i]

                # Exact Match Accuracy
                if predicted_answer_cleaned.lower() == true_answer_string.lower():
                    correct_exact_matches += 1

                # Numerical Accuracy (Extract numbers from both predicted and true)
                pred_numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', predicted_answer_cleaned)]
                true_numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', true_answer_string)]

                if len(pred_numbers) > 0 and len(true_numbers) > 0:
                    if math.isclose(pred_numbers[0], true_numbers[0], rel_tol=1e-5, abs_tol=1e-5):
                        correct_numerical_matches += 1
                elif len(pred_numbers) == 0 and len(true_numbers) == 0:
                    correct_numerical_matches += 1


    exact_accuracy = correct_exact_matches / total_examples if total_examples > 0 else 0.0
    numerical_accuracy = correct_numerical_matches / total_examples if total_examples > 0 else 0.0

    return exact_accuracy, numerical_accuracy
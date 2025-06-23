# File: blackhole/nova/training.py

import torch
from tqdm import tqdm
# Remove: from torch.utils.data import DataLoader  (not needed here for train_step)

# Correct relative import for loss functions
# Assuming loss_functions.py is in the same 'nova' directory
from .loss_functions import focal_loss, mse_loss_for_numerical_features

# Remove this line, as predict_and_decode_answer is not used in train_step
# from blackhole.nova.prediction import predict_and_decode_answer

def train_step(model, batch, optimizer, scheduler, focal_loss_fn, mse_loss_fn, vocab, device, clip_grad=1.0):
    model.train()

    # Move batch tensors to the specified device
    encoder_token_ids = batch['encoder_token_ids'].to(device)
    encoder_numeric_features = batch['encoder_numeric_features'].to(device)
    encoder_attention_mask = batch['encoder_attention_mask'].to(device)
    decoder_input_token_ids = batch['decoder_input_token_ids'].to(device)
    decoder_input_numeric_features = batch['decoder_input_numeric_features'].to(device)
    decoder_output_token_targets = batch['decoder_output_token_targets'].to(device)
    decoder_output_numeric_targets = batch['decoder_output_numeric_targets'].to(device)
    decoder_attention_mask = batch['decoder_attention_mask'].to(device)

    optimizer.zero_grad()

    # Forward pass through the model
    token_logits, num_feature_output = model(
        encoder_token_ids=encoder_token_ids,
        encoder_numeric_features=encoder_numeric_features,
        encoder_attention_mask=encoder_attention_mask,
        decoder_token_ids=decoder_input_token_ids,
        decoder_numeric_features_input=decoder_input_numeric_features,
        decoder_attention_mask=decoder_attention_mask
    )

    # Get special token IDs for loss calculation
    pad_id = vocab.get('<|pad|>', 0)
    num_id = vocab.get('<|num|>', vocab.get('<|unk|>', 0)) # Fallback for num_id if not found

    # Calculate classification loss (Focal Loss for tokens)
    ce_loss = focal_loss_fn(token_logits.view(-1, token_logits.size(-1)),
                             decoder_output_token_targets.view(-1),
                             ignore_index=pad_id)

    # Calculate numerical features loss (MSE)
    num_target_mask = (decoder_output_token_targets == num_id)
    numeric_loss = mse_loss_fn(
        num_feature_output,
        decoder_output_numeric_targets,
        num_target_mask
    )

    # Total loss: Adjust weights as needed
    loss = ce_loss + 0.5 * numeric_loss # Example: Increased weight for numeric loss

    # Backpropagation and optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Return only the total loss tensor
    return loss
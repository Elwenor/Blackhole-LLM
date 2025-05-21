import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import loss functions from the same package
from blackhole.nova.loss_functions import focal_loss, mse_loss_for_numerical_features
# Import prediction logic from the same package
from blackhole.nova.prediction import predict_and_decode_answer # We'll define this next

def train_step(model, optimizer, scheduler, batch, vocab, device, clip_grad=1.0):
    """
    Performs one training step for the model.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        batch (dict): A dictionary containing input and target tensors for the batch.
        vocab (dict): The vocabulary mapping tokens to IDs.
        device (torch.device): The device (CPU/GPU) to perform computations on.
        clip_grad (float): Maximum gradient norm for gradient clipping.

    Returns:
        dict: A dictionary containing the computed losses for the step.
    """
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
    ce_loss = focal_loss(token_logits.view(-1, token_logits.size(-1)),
                         decoder_output_token_targets.view(-1),
                         ignore_index=pad_id)

    # Calculate numerical features loss (MSE)
    num_target_mask = (decoder_output_token_targets == num_id)
    numeric_loss = mse_loss_for_numerical_features(
        num_feature_output,
        decoder_output_numeric_targets,
        num_target_mask
    )

    # Total loss: Adjust weights as needed
    loss = ce_loss + 0.5 * numeric_loss # Example: Increased weight for numeric loss

    # Backpropagation and optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad) # Clip gradients to prevent exploding gradients
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

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataset (torch.utils.data.Dataset): The dataset for evaluation.
        vocab (dict): The vocabulary mapping tokens to IDs.
        device (torch.device): The device (CPU/GPU) for computations.
        batch_size (int): Batch size for evaluation.
        max_decoding_len (int): Maximum length for generated sequences.
        collate_fn (callable, optional): Custom collate function for the DataLoader.

    Returns:
        tuple: A tuple containing (exact_accuracy, numerical_accuracy).
    """
    model.eval()
    idx_to_token = {idx: token for token, idx in vocab.items()}
    num_token_id = vocab.get('<|num|>', vocab.get('<|unk|>', 0))
    bos_token_id = vocab.get('<|bos|>', vocab.get('<|unk|>', 0))
    eos_token_id = vocab.get('<|eos|>', vocab.get('<|unk|>', 0))
    pad_token_id = vocab.get('<|pad|>', 0)

    # Initialize a padded feature row for numeric features when not present
    # This assumes 'model.feature_dim' is accessible from the model instance
    padded_feat_row = torch.full((model.feature_dim,), -2.0, dtype=torch.float32).to(device)

    correct_exact_matches = 0
    correct_numerical_matches = 0
    total_examples = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move encoder inputs to device
            encoder_token_ids = batch['encoder_token_ids'].to(device)
            encoder_numeric_features = batch['encoder_numeric_features'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)

            batch_size_actual = encoder_token_ids.size(0)

            # Call the shared prediction function
            predicted_answer_strings = predict_and_decode_answer(
                model=model,
                encoder_token_ids=encoder_token_ids,
                encoder_numeric_features=encoder_numeric_features,
                encoder_attention_mask=encoder_attention_mask,
                vocab=vocab,
                device=device,
                max_decoding_len=max_decoding_len,
                padded_feat_row=padded_feat_row # Pass the padded_feat_row
            )

            # Calculate accuracy for the batch
            for i in range(batch_size_actual):
                total_examples += 1
                true_answer_string = batch['original_answer_strings'][i]
                predicted_answer_cleaned = predicted_answer_strings[i]

                # Exact Match Accuracy (case-insensitive)
                if predicted_answer_cleaned.lower() == true_answer_string.lower():
                    correct_exact_matches += 1

                # Numerical Accuracy (Extract numbers from both predicted and true)
                # Using regex to find all numbers (integers and floats, positive/negative)
                pred_numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', predicted_answer_cleaned)]
                true_numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', true_answer_string)]

                # Compare only if both have at least one number and they are 'close'
                if len(pred_numbers) > 0 and len(true_numbers) > 0:
                    # Only compare the first number for simplicity; extend as needed for multiple numbers
                    if math.isclose(pred_numbers[0], true_numbers[0], rel_tol=1e-5, abs_tol=1e-5):
                        correct_numerical_matches += 1
                elif len(pred_numbers) == 0 and len(true_numbers) == 0:
                    # Both have no numbers, consider it a match
                    correct_numerical_matches += 1

    exact_accuracy = correct_exact_matches / total_examples if total_examples > 0 else 0.0
    numerical_accuracy = correct_numerical_matches / total_examples if total_examples > 0 else 0.0

    return exact_accuracy, numerical_accuracy
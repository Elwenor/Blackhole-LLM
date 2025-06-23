# File: blackhole/nova/evaluation.py

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .prediction import predict_and_decode_answer # Assuming predict_and_decode_answer is in blackhole/nova/prediction.py
from blackhole.embedding import decode_number_from_features
import re
import math # Import math for isnan

def evaluate(model, dataset, vocab, device, batch_size, max_decoding_len, collate_fn):
    model.eval()
    eval_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    total_exact_matches = 0
    total_numerical_matches = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            encoder_token_ids = batch['encoder_token_ids'].to(device)
            encoder_numeric_features = batch['encoder_numeric_features'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            # target_answer_tokens = batch['decoder_output_token_targets'] # On CPU for string comparison - Not directly used here
            target_numeric_values_batch = batch['original_numeric_values'] # On CPU for comparison
            original_answers_batch = batch['original_answer'] # Get original answers from batch

            # Generate predictions - Now expects two lists
            predicted_texts_batch, predicted_numbers_batch = predict_and_decode_answer(
                model,
                encoder_token_ids,
                encoder_numeric_features,
                encoder_attention_mask,
                vocab,
                device,
                max_decoding_len
            )
            
            for i in range(len(predicted_texts_batch)):
                total_examples += 1
                predicted_text = predicted_texts_batch[i]
                predicted_number = predicted_numbers_batch[i] # Use the directly predicted number

                # Get the true answer and true numeric value from the batch
                true_answer_text_full = original_answers_batch[i] # Get from the batch
                
                # MODIFIED: Safely extract the first true numeric value from the tensor
                if target_numeric_values_batch[i].numel() > 0 and target_numeric_values_batch[i][0].item() != -1.0:
                    true_answer_num_match = target_numeric_values_batch[i][0].item()
                else:
                    true_answer_num_match = None


                # Handle NaN for numerical comparisons:
                if true_answer_num_match is not None and math.isnan(true_answer_num_match):
                    true_answer_num_match = None


                # Exact Match Accuracy (simplified comparison)
                if normalize_text(predicted_text) == normalize_text(true_answer_text_full):
                    total_exact_matches += 1

                # Numerical Accuracy (comparing the directly predicted number with the ground truth)
                if true_answer_num_match is not None and predicted_number is not None:
                    if abs(true_answer_num_match - predicted_number) < 1e-3: # Tolerance for float comparison
                        total_numerical_matches += 1

    exact_match_accuracy = total_exact_matches / total_examples if total_examples > 0 else 0.0
    numerical_accuracy = total_numerical_matches / total_examples if total_examples > 0 else 0.0

    return exact_match_accuracy, numerical_accuracy


def extract_final_answer_number(text):
    """Extracts the final numerical answer from a GSM8K-like text."""
    # This function is now less critical for numerical accuracy if we directly get predicted_number.
    # It might still be useful for fallback or specific text-based extraction.
    match = re.search(r'(?:(?:the\s+answer\s+is|is)\s+)?(\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    
    # Fallback: try to find any standalone number or number at the end
    numbers = re.findall(r'(\d+\.?\d*)', text)
    if numbers:
        try:
            return float(numbers[-1]) # Take the last number found
        except ValueError:
            return None
    return None

def normalize_text(text):
    """Basic text normalization for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single
    text = re.sub(r'[^a-z0-9\s\.]', '', text) # Keep only alphanumeric, spaces, dot
    return text
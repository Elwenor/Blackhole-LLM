# File: blackhole/nova/evaluation.py

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .prediction import predict_and_decode_answer # Assuming predict_and_decode_answer is in blackhole/nova/prediction.py
from blackhole.embedding import decode_number_from_features
import re

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
            target_answer_tokens = batch['decoder_output_token_targets'] # On CPU for string comparison
            target_numeric_values = batch['original_numeric_values'] # On CPU for comparison

            # Generate predictions
            predicted_answers_decoded = predict_and_decode_answer(
                model,
                encoder_token_ids,
                encoder_numeric_features,
                encoder_attention_mask,
                vocab,
                device,
                max_decoding_len
            )
            

            for i in range(len(predicted_answers_decoded)):
                total_examples += 1
                predicted_text = predicted_answers_decoded[i]
                
                try:
                    true_answer_text_full = dataset[batch['original_indices'][i]]['answer']
                except Exception:
                    # Fallback for dummy dataset or if original_indices is not available
                    # This implies you must have 'original_answers_text' in your batch or dataset mapping.
                    true_answer_text_full = "Dummy Answer 123" # REPLACE WITH ACTUAL GROUND TRUTH

                true_answer_num_match = extract_final_answer_number(true_answer_text_full)
                predicted_answer_num_match = extract_final_answer_number(predicted_text)

                # Exact Match Accuracy (simplified comparison)
                # This comparison is tricky due to tokenization and numerical representation.
                # A robust exact match would require careful normalization of both true and predicted text.
                if normalize_text(predicted_text) == normalize_text(true_answer_text_full):
                    total_exact_matches += 1

                # Numerical Accuracy (comparing the final extracted numbers)
                if true_answer_num_match is not None and predicted_answer_num_match is not None:
                    if abs(true_answer_num_match - predicted_answer_num_match) < 1e-3: # Tolerance for float comparison
                        total_numerical_matches += 1

    exact_match_accuracy = total_exact_matches / total_examples if total_examples > 0 else 0.0
    numerical_accuracy = total_numerical_matches / total_examples if total_examples > 0 else 0.0

    return exact_match_accuracy, numerical_accuracy


def extract_final_answer_number(text):
    """Extracts the final numerical answer from a GSM8K-like text."""
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
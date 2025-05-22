# File: data_processing.py (Excerpt - you will need to add this logic)

import re
from num2words import num2words
import torch
import numpy as np
from datasets import Features, Value, Sequence
import sys, os

# Assuming these are defined in your config.py or elsewhere
from config import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN, SPECIAL_TOKENS
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blackhole.embedding import *

def get_features_schema(numeric_feature_dim, max_seq_len):
    """Defines the features schema for the dataset."""
    return Features({
        'encoder_token_ids': Sequence(Value('int32'), length=max_seq_len),
        'encoder_numeric_features': Sequence(Sequence(Value('float32'), length=numeric_feature_dim), length=max_seq_len),
        'encoder_attention_mask': Sequence(Value('bool'), length=max_seq_len),
        'decoder_input_token_ids': Sequence(Value('int32'), length=max_seq_len),
        'decoder_input_numeric_features': Sequence(Sequence(Value('float32'), length=numeric_feature_dim), length=max_seq_len),
        'decoder_output_token_targets': Sequence(Value('int32'), length=max_seq_len),
        'decoder_output_numeric_targets': Sequence(Sequence(Value('float32'), length=numeric_feature_dim), length=max_seq_len),
        'decoder_attention_mask': Sequence(Value('bool'), length=max_seq_len),
        # ADD THESE TWO NEW FIELDS:
        'original_answers_text': Value('string'), # To store the raw answer string
        'original_numeric_values': Sequence(Value('float32')) # To store all numbers extracted from the answer
    })

def preprocess_example(example, vocab, numeric_feature_dim, max_seq_len):
    question = example['question']
    answer = example['answer']

    # --- Process Question (Encoder Input) ---
    encoder_tokens, encoder_nums, encoder_num_types = tokenize_and_featurize_text(
        question, NUM_TOKEN, PAD_TOKEN
    )
    encoder_token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in encoder_tokens]
    encoder_token_ids = [vocab[BOS_TOKEN]] + encoder_token_ids + [vocab[EOS_TOKEN]]
    
    encoder_numeric_features = [
        number_embedding_features(val, typ) for val, typ in zip(encoder_nums, encoder_num_types)
    ]
    # Add padding for BOS/EOS tokens for numeric features (e.g., all zeros or a special "no number" embedding)
    padded_feat_row = [0.0] * numeric_feature_dim # Or a specific padding value like -2.0
    encoder_numeric_features = [padded_feat_row] + encoder_numeric_features + [padded_feat_row]

    encoder_attention_mask = [True] * len(encoder_token_ids)

    # Pad/truncate encoder inputs
    encoder_token_ids, encoder_numeric_features, encoder_attention_mask = pad_sequence_for_model(
        encoder_token_ids, encoder_numeric_features, encoder_attention_mask,
        max_seq_len, vocab[PAD_TOKEN], padded_feat_row, is_encoder=True
    )

    # --- Process Answer (Decoder Input/Output) ---
    decoder_tokens, decoder_nums, decoder_num_types = tokenize_and_featurize_text(
        answer, NUM_TOKEN, PAD_TOKEN
    )
    
    # Decoder input: starts with BOS, then tokens from answer
    decoder_input_token_ids = [vocab[BOS_TOKEN]] + [vocab.get(token, vocab[UNK_TOKEN]) for token in decoder_tokens]
    
    decoder_input_numeric_features = [padded_feat_row] + [ # BOS gets padding
        number_embedding_features(val, typ) for val, typ in zip(decoder_nums, decoder_num_types)
    ]

    # Decoder output targets: tokens from answer, then EOS
    decoder_output_token_targets = [vocab.get(token, vocab[UNK_TOKEN]) for token in decoder_tokens] + [vocab[EOS_TOKEN]]

    decoder_output_numeric_targets = [
        number_embedding_features(val, typ) for val, typ in zip(decoder_nums, decoder_num_types)
    ] + [padded_feat_row] # EOS gets padding

    decoder_attention_mask = [True] * len(decoder_input_token_ids) # Assuming it's `True` for valid tokens

    # Pad/truncate decoder inputs/outputs
    decoder_input_token_ids, decoder_input_numeric_features, decoder_attention_mask = pad_sequence_for_model(
        decoder_input_token_ids, decoder_input_numeric_features, decoder_attention_mask,
        max_seq_len, vocab[PAD_TOKEN], padded_feat_row, is_encoder=False
    )
    decoder_output_token_targets, decoder_output_numeric_targets, _ = pad_sequence_for_model(
        decoder_output_token_targets, decoder_output_numeric_targets, [True] * len(decoder_output_token_targets),
        max_seq_len, vocab[PAD_TOKEN], padded_feat_row, is_encoder=False
    )
    
    # Extract all numbers from the original answer text
    # This is the crucial part for 'original_numeric_values'
    original_numbers_in_answer = [float(n) for n in re.findall(r'\d+\.?\d*', answer)]
    if not original_numbers_in_answer:
        original_numbers_in_answer = [0.0] # Ensure it's never empty, provide a default if no numbers are found

    return {
        'encoder_token_ids': encoder_token_ids,
        'encoder_numeric_features': encoder_numeric_features,
        'encoder_attention_mask': encoder_attention_mask,
        'decoder_input_token_ids': decoder_input_token_ids,
        'decoder_input_numeric_features': decoder_input_numeric_features,
        'decoder_output_token_targets': decoder_output_token_targets,
        'decoder_output_numeric_targets': decoder_output_numeric_targets,
        'decoder_attention_mask': decoder_attention_mask,
        'original_answers_text': answer, # Store the original answer text
        'original_numeric_values': original_numbers_in_answer # Store the list of numbers from the original answer
    }


def custom_collate_fn(batch, vocab, numeric_feature_dim, max_seq_len):
    # This function needs to handle the newly added 'original_answers_text' and 'original_numeric_values'
    # It should stack them correctly.
    
    # Extract lists of all fields from the batch
    encoder_token_ids_list = [item['encoder_token_ids'] for item in batch]
    encoder_numeric_features_list = [item['encoder_numeric_features'] for item in batch]
    encoder_attention_mask_list = [item['encoder_attention_mask'] for item in batch]
    decoder_input_token_ids_list = [item['decoder_input_token_ids'] for item in batch]
    decoder_input_numeric_features_list = [item['decoder_input_numeric_features'] for item in batch]
    decoder_output_token_targets_list = [item['decoder_output_token_targets'] for item in batch]
    decoder_output_numeric_targets_list = [item['decoder_output_numeric_targets'] for item in batch]
    decoder_attention_mask_list = [item['decoder_attention_mask'] for item in batch]
    
    # NEW: Extract these lists
    original_answers_text_list = [item['original_answers_text'] for item in batch]
    original_numeric_values_list = [item['original_numeric_values'] for item in batch]

    # Convert to tensors
    padded_encoder_token_ids = torch.tensor(encoder_token_ids_list, dtype=torch.long)
    padded_encoder_numeric_features = torch.tensor(encoder_numeric_features_list, dtype=torch.float32)
    padded_encoder_attention_mask = torch.tensor(encoder_attention_mask_list, dtype=torch.bool)
    padded_decoder_input_token_ids = torch.tensor(decoder_input_token_ids_list, dtype=torch.long)
    padded_decoder_input_numeric_features = torch.tensor(decoder_input_numeric_features_list, dtype=torch.float32)
    padded_decoder_output_token_targets = torch.tensor(decoder_output_token_targets_list, dtype=torch.long)
    padded_decoder_output_numeric_targets = torch.tensor(decoder_output_numeric_targets_list, dtype=torch.float32)
    padded_decoder_attention_mask = torch.tensor(decoder_attention_mask_list, dtype=torch.bool)

    # For original_numeric_values, you might have variable lengths.
    # You need to handle this. One common way is to pad them or store them as a list of lists.
    # If `evaluate` expects a single tensor, you'll need a padding strategy.
    # For now, let's keep them as a list of lists if `evaluate` can handle that,
    # or pad them to max_seq_len if they're used in a vectorized way for numerical accuracy.
    # Given the current `evaluate` function, it's looping through each item, so a list of lists is fine.
    
    # You might also want to store original_indices if you need to fetch from the raw dataset later
    # original_indices = [item['original_index'] for item in batch] # If you added this in preprocess

    return {
        'encoder_token_ids': padded_encoder_token_ids,
        'encoder_numeric_features': padded_encoder_numeric_features,
        'encoder_attention_mask': padded_encoder_attention_mask,
        'decoder_input_token_ids': padded_decoder_input_token_ids,
        'decoder_input_numeric_features': padded_decoder_input_numeric_features,
        'decoder_output_token_targets': padded_decoder_output_token_targets,
        'decoder_output_numeric_targets': padded_decoder_output_numeric_targets,
        'decoder_attention_mask': padded_decoder_attention_mask,
        'original_answers_text': original_answers_text_list, # Pass as list of strings
        'original_numeric_values': original_numeric_values_list, # Pass as list of lists of floats
        # 'original_indices': original_indices, # If you added original_index in preprocess
    }

# Ensure `tokenize_and_featurize_text` and `pad_sequence_for_model` are correctly defined
# in data_processing.py or imported from relevant modules.
# Example of a simplified tokenize_and_featurize_text (you have your own, ensure it works with numbers)
def tokenize_and_featurize_text(text, num_token, pad_token):
    # This is a placeholder. Your actual implementation should extract numbers and tokens.
    tokens = []
    numbers = []
    num_types = []
    
    # Replace numbers with <|num|> token and store the actual number
    parts = re.split(r'(\d+\.?\d*)', text)
    for part in parts:
        if re.fullmatch(r'\d+\.?\d*', part):
            tokens.append(num_token)
            numbers.append(float(part))
            num_types.append('float' if '.' in part else 'int')
        elif part: # Process non-numeric parts
            sub_tokens = re.findall(r"[\w']+|[.,!?;:]|\s+", part.lower())
            for tok in sub_tokens:
                if tok.isspace():
                    if tokens and tokens[-1] != ' ': # Avoid multiple spaces
                        tokens.append(' ')
                else:
                    tokens.append(tok)
            # Add dummy numbers and types for non-numeric tokens
            numbers.extend([0.0] * len([t for t in sub_tokens if not t.isspace()]))
            num_types.extend(['float'] * len([t for t in sub_tokens if not t.isspace()]))
    
    return tokens, numbers, num_types

def pad_sequence_for_model(token_ids, numeric_features, attention_mask, max_len, pad_id, padded_feat_row, is_encoder=True):
    # This is a placeholder. Your actual implementation.
    current_len = len(token_ids)
    
    if current_len > max_len:
        if is_encoder: # Truncate from the end
            token_ids = token_ids[:max_len]
            numeric_features = numeric_features[:max_len]
            attention_mask = attention_mask[:max_len]
        else: # Truncate from the end for decoder as well, but be careful with EOS
            token_ids = token_ids[:max_len]
            numeric_features = numeric_features[:max_len]
            attention_mask = attention_mask[:max_len]
            # Ensure EOS is at max_len if it was there and got truncated
            if token_ids[-1] != pad_id and token_ids[-1] != vocab.get(EOS_TOKEN):
                # Optionally replace last token with EOS if it was truncated, but keep logic simple for now
                pass 
    elif current_len < max_len:
        padding_len = max_len - current_len
        token_ids.extend([pad_id] * padding_len)
        numeric_features.extend([padded_feat_row] * padding_len)
        attention_mask.extend([False] * padding_len) # False for masked (padded) tokens

    return token_ids, numeric_features, attention_mask
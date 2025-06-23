# File: blackhole/nova/data_processing.py

import sys
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from num2words import num2words
import re
from collections import Counter
from transformers import AutoTokenizer

# Add the directory of the current script to sys.path first for local imports
# This ensures that sibling modules like 'training', 'evaluation', etc., are found.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set base_dir for blackhole imports (project root: C:\Users\Aleksander\Documents\GitHub\Blackhole-LLM)
try:
    # This path is already correct to go from 'scripts/training' to 'Blackhole-LLm'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
except NameError:
    base_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))

# Add base_dir to sys.path if it's not already there (should be distinct from script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir) # Insert at the beginning to prioritize blackhole package imports

from config import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN,
    CAP_TOKEN, ALLCAPS_TOKEN, SPACE_TOKEN, SPECIAL_TOKENS,
    MAX_SEQ_LEN
)
from datasets import Value, Features, Sequence


def get_features_schema(determined_numeric_feature_dim):
    """
    Returns the features schema for the dataset.
    """
    return Features({
        'encoder_token_ids': Sequence(Value('int32')),
        'encoder_numeric_features': Sequence(Sequence(Value('float32'))),
        'encoder_attention_mask': Sequence(Value('bool')),
        'decoder_input_token_ids': Sequence(Value('int32')),
        'decoder_input_numeric_features': Sequence(Sequence(Value('float32'))),
        'decoder_output_token_targets': Sequence(Value('int32')),
        'decoder_output_numeric_targets': Sequence(Sequence(Value('float32'))),
        'decoder_attention_mask': Sequence(Value('bool')),
        'original_question': Value('string'),
        'original_answer': Value('string'),
        'original_numeric_values': Sequence(Value('float32')),
        'answer_token_type_map': Sequence(Value('string')) # Added for debugging
    })

def create_number_features_tensor(value: float, num_dim: int) -> list:
    """
    Creates a numerical feature vector for a given number.
    This is a placeholder and can be expanded for more sophisticated number representations.
    """
    features = [0.0] * num_dim
    if num_dim > 0:
        features[0] = float(value) # Store the value itself
    if num_dim > 1:
        features[1] = math.log(abs(value)) if value != 0 else 0.0 # Log scale
    if num_dim > 2:
        features[2] = 1.0 if value < 0 else 0.0 # Sign
    return features

def number_embedding_features(value: float, typ: str) -> list:
    """
    Generates rich features for numerical values based on their type (int/float).
    The 'typ' argument helps in differentiating how integers and floats are processed.
    """
    features = [0.0] * 7 # Ensure this matches determined_numeric_feature_dim in config or is dynamic

    # Feature 0: The number itself
    features[0] = value

    # Feature 1: Logarithm of the absolute value (for scale)
    features[1] = math.log(abs(value)) if value != 0 else 0.0

    # Feature 2: Sign (1 for negative, 0 for non-negative)
    features[2] = 1.0 if value < 0 else 0.0

    # Feature 3: Is Integer?
    features[3] = 1.0 if typ == 'int' else 0.0

    # Feature 4: Is Float?
    features[4] = 1.0 if typ == 'float' else 0.0

    # Feature 5: Value mapped to a periodic function (e.g., sine for cyclic patterns)
    features[5] = math.sin(value)

    # Feature 6: Value mapped to another periodic function (e.g., cosine for cyclic patterns)
    features[6] = math.cos(value)

    # Ensure no NaNs or Infs, replace with 0 if they occur
    features = [f if math.isfinite(f) else 0.0 for f in features]

    return features


def tokenize(text, special_tokens=None):
    if special_tokens is None:
        special_tokens = []

    tokens = []
    number_map = {}
    last_end = 0
    num_placeholder_count = 0

    # Pattern to find numbers (integers or floats, optionally with sign)
    number_pattern = r"[-+]?\d*\.\d+|\d+"

    for m in re.finditer(number_pattern, text):
        # Add pre-number text
        pre_num_text = text[last_end:m.start()]
        if pre_num_text:
            # Simple word and non-word tokenizer for text segments
            pre_num_tokens = re.findall(r'\b\w+\b|[^s\w\s]', pre_num_text)
            for t in pre_num_tokens:
                if t.strip(): # Ensure token is not just whitespace
                    tokens.append(t)

        # Add NUM_TOKEN and map the number
        num_str = m.group(0)
        try:
            num_val = float(num_str)
            tokens.append(NUM_TOKEN)
            # Store the actual float value, its inferred type, and original string
            number_map[len(tokens) - 1] = (num_val, 'float' if '.' in num_str or 'e' in num_str.lower() else 'int', num_str)
        except ValueError:
            # Fallback if conversion to float fails (e.g., very malformed number)
            tokens.append(num_str) # Treat as a regular token if it can't be a number
        last_end = m.end()

    # Add remaining text
    remaining_text = text[last_end:]
    if remaining_text:
        remaining_tokens = re.findall(r'\b\w+\b|[^s\w\s]', remaining_text)
        for t in remaining_tokens:
            if t.strip():
                tokens.append(t)

    # Process for capitalization and special tokens
    final_tokens = []
    for token in tokens:
        if token in special_tokens:
            final_tokens.append(token)
        elif token.isupper() and len(token) > 1 and token not in SPECIAL_TOKENS:
            final_tokens.append(ALLCAPS_TOKEN)
            final_tokens.append(token.lower())
        elif token.istitle() and token not in SPECIAL_TOKENS:
            final_tokens.append(CAP_TOKEN)
            final_tokens.append(token.lower())
        elif token == ' ': # Replace standalone spaces if needed, or handle differently
            final_tokens.append(SPACE_TOKEN)
        else:
            final_tokens.append(token)
    return final_tokens, number_map


def build_vocabulary(train_dataset, eval_dataset, min_freq=1):
    """
    Builds a vocabulary from the training and evaluation datasets,
    including special tokens and handling numbers.
    """
    word_counts = Counter()

    # First, gather all tokens from questions and answers
    for dataset in [train_dataset, eval_dataset]:
        for example in dataset:
            question_tokens, _ = tokenize(example['question'], SPECIAL_TOKENS)
            word_counts.update(question_tokens)

            if 'answers' in example and example['answers'] and 'text' in example['answers'] and len(example['answers']['text']) > 0:
                answer_tokens, _ = tokenize(example['answers']['text'][0], SPECIAL_TOKENS)
                word_counts.update(answer_tokens)

    # Add special tokens
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3, NUM_TOKEN: 4, CAP_TOKEN: 5, ALLCAPS_TOKEN: 6, SPACE_TOKEN: 7}
    idx = len(vocab)

    # Add words based on frequency
    for word, count in word_counts.items():
        if word not in vocab and count >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def preprocess_example(example, vocab, num_dim, max_seq_len, idx_to_token):
    question = example['question'][0]
    answer = example['answer'][0] if example['answer'] else ""

    # Tokenize question
    encoder_tokens, encoder_number_map = tokenize(question, SPECIAL_TOKENS)
    encoder_token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in encoder_tokens]
    # Initialize encoder_numeric_features with default for each token
    encoder_numeric_features = [number_embedding_features(0.0, 'float') for _ in encoder_token_ids]

    # Populate actual numeric features for NUM_TOKENs
    for idx, (value, typ, original_str) in encoder_number_map.items():
        if idx < len(encoder_numeric_features): # Ensure index is within bounds
            encoder_numeric_features[idx] = number_embedding_features(value, typ)


    # Tokenize answer for decoder input and target
    decoder_output_tokens, decoder_number_map = tokenize(answer, SPECIAL_TOKENS)

    # Decoder input will have BOS, followed by answer tokens
    decoder_input_tokens = [BOS_TOKEN] + decoder_output_tokens
    decoder_input_token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in decoder_input_tokens]

    # Decoder output targets will be answer tokens followed by EOS
    decoder_output_token_targets = [vocab.get(token, vocab[UNK_TOKEN]) for token in decoder_output_tokens] + [vocab[EOS_TOKEN]]

    # Initialize decoder_input_numeric_features with default for BOS and other tokens
    decoder_input_numeric_features = [number_embedding_features(0.0, 'float')] + [number_embedding_features(0.0, 'float') for _ in decoder_output_tokens]
    decoder_output_numeric_targets = [number_embedding_features(0.0, 'float') for _ in decoder_output_tokens] + [number_embedding_features(0.0, 'float')]

    # Populate actual numeric features for NUM_TOKENs in decoder sequences
    original_numeric_values = [] # Store original numeric values from the answer
    answer_token_type_map = [] # To track which tokens are numbers or text in the answer

    for i, token in enumerate(decoder_output_tokens):
        if token == NUM_TOKEN and i in decoder_number_map:
            value, typ, original_str = decoder_number_map[i]
            if i + 1 < len(decoder_input_numeric_features): # +1 because of BOS token in input
                decoder_input_numeric_features[i + 1] = number_embedding_features(value, typ)
            if i < len(decoder_output_numeric_targets):
                decoder_output_numeric_targets[i] = number_embedding_features(value, typ)
            original_numeric_values.append(value)
            answer_token_type_map.append('NUM')
        else:
            answer_token_type_map.append('TEXT')
            # For non-numeric tokens, ensure a consistent feature vector (e.g., all zeros or a special "no number" embedding)
            # This is already handled by initialization with number_embedding_features(0.0, 'float')

    # Pad and truncate sequences
    def pad_sequence(ids, features, max_len, pad_id, padded_feat_row):
        current_len = len(ids)
        if current_len > max_len:
            return ids[:max_len], features[:max_len], [True] * max_len # Truncate and all attention True
        else:
            padding_len = max_len - current_len
            padded_ids = ids + [pad_id] * padding_len
            padded_features = features + [padded_feat_row] * padding_len
            attention_mask = [True] * current_len + [False] * padding_len
            return padded_ids, padded_features, attention_mask

    padded_feat_row = number_embedding_features(-2.0, 'float') # Use a distinct value for padding numeric features

    encoder_token_ids, encoder_numeric_features, encoder_attention_mask = pad_sequence(
        encoder_token_ids, encoder_numeric_features, max_seq_len, vocab[PAD_TOKEN], padded_feat_row
    )
    decoder_input_token_ids, decoder_input_numeric_features, decoder_input_attention_mask = pad_sequence(
        decoder_input_token_ids, decoder_input_numeric_features, max_seq_len, vocab[PAD_TOKEN], padded_feat_row
    )
    decoder_output_token_targets, decoder_output_numeric_targets, decoder_output_attention_mask = pad_sequence(
        decoder_output_token_targets, decoder_output_numeric_targets, max_seq_len, vocab[PAD_TOKEN], padded_feat_row
    )

    # For original_numeric_values, ensure it's a list even if empty
    if not original_numeric_values:
        original_numeric_values = [-1.0] # Placeholder if no numbers, assuming -1.0 is safe

    return {
        'encoder_token_ids': encoder_token_ids,
        'encoder_numeric_features': encoder_numeric_features,
        'encoder_attention_mask': encoder_attention_mask,
        'decoder_input_token_ids': decoder_input_token_ids,
        'decoder_input_numeric_features': decoder_input_numeric_features,
        'decoder_output_token_targets': decoder_output_token_targets,
        'decoder_output_numeric_targets': decoder_output_numeric_targets,
        'decoder_attention_mask': decoder_output_attention_mask, # Should be decoder_output_attention_mask
        'original_question': question,
        'original_answer': answer,
        'original_numeric_values': original_numeric_values,
        'answer_token_type_map': answer_token_type_map
    }


def custom_collate_fn(batch, pad_token_id, padded_feat_row, enc_max_seq_len, dec_max_seq_len):
    """
    Custom collate function for the DataLoader to handle stacking pre-processed tensors.
    """
    # These items are already expected to be torch.Tensor due to dataset.set_format(type='torch')
    encoder_token_ids = torch.stack([item['encoder_token_ids'] for item in batch])
    encoder_numeric_features = torch.stack([item['encoder_numeric_features'] for item in batch])
    encoder_attention_mask = torch.stack([item['encoder_attention_mask'] for item in batch])

    decoder_input_token_ids = torch.stack([item['decoder_input_token_ids'] for item in batch])
    decoder_input_numeric_features = torch.stack([item['decoder_input_numeric_features'] for item in batch])
    decoder_attention_mask = torch.stack([item['decoder_attention_mask'] for item in batch])
    decoder_output_token_targets = torch.stack([item['decoder_output_token_targets'] for item in batch])
    decoder_output_numeric_targets = torch.stack([item['decoder_output_numeric_targets'] for item in batch])

    # Keep these as lists as they are auxiliary outputs and may have variable lengths within the batch
    original_numeric_values = [item['original_numeric_values'] for item in batch]
    answer_token_type_map = [item['answer_token_type_map'] for item in batch]
    
    # Use .get() with a default value to prevent KeyError if these fields are occasionally missing
    original_question = [item.get('original_question', "") for item in batch]
    original_answer = [item.get('original_answer', "") for item in batch]

    return {
        'encoder_token_ids': encoder_token_ids,
        'encoder_numeric_features': encoder_numeric_features,
        'encoder_attention_mask': encoder_attention_mask,
        'decoder_input_token_ids': decoder_input_token_ids,
        'decoder_input_numeric_features': decoder_input_numeric_features,
        'decoder_output_token_targets': decoder_output_token_targets,
        'decoder_output_numeric_targets': decoder_output_numeric_targets,
        'decoder_attention_mask': decoder_attention_mask,
        'original_numeric_values': original_numeric_values,
        'answer_token_type_map': answer_token_type_map,
        'original_question': original_question,
        'original_answer': original_answer
    }
# File: data_processing.py

import re
from num2words import num2words
import torch
import numpy as np
from datasets import Features, Value, Sequence
import sys, os
from collections import Counter

from config import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN, SPECIAL_TOKENS,
    CAP_TOKEN, ALLCAPS_TOKEN, SPACE_TOKEN
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
        'original_answers_text': Value('string'),
        'original_numeric_values': Sequence(Value('float32'))
    })

def basic_tokenize(text):
    tokens = re.findall(r"[\w']+|[.,!?;:()]|\s+", text.lower())
    final_tokens = []
    for tok in tokens:
        if tok.isspace():
            if final_tokens and final_tokens[-1] != SPACE_TOKEN:
                final_tokens.append(SPACE_TOKEN)
        elif re.fullmatch(r'\d+\.?\d*', tok):
            final_tokens.append(NUM_TOKEN)
        elif tok in [CAP_TOKEN.lower(), ALLCAPS_TOKEN.lower()]:
             final_tokens.append(tok)
        else:
            final_tokens.append(tok)
            
    if final_tokens and final_tokens[0] == SPACE_TOKEN:
        final_tokens = final_tokens[1:]
    if final_tokens and final_tokens[-1] == SPACE_TOKEN:
        final_tokens = final_tokens[:-1]

    return final_tokens

def build_vocab_from_dataset(dataset_dict):
    token_counts = Counter()
    for split_name in dataset_dict:
        current_dataset = dataset_dict[split_name]
        for example in current_dataset:
            question = example.get('question', '')
            answer = example.get('answer', '')
            if question:
                tokens = basic_tokenize(question)
                token_counts.update(tokens)
            if answer:
                tokens = basic_tokenize(answer)
                token_counts.update(tokens)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        BOS_TOKEN: 2,
        EOS_TOKEN: 3,
        NUM_TOKEN: 4,
    }
    
    next_id = len(vocab)
    for token in SPECIAL_TOKENS:
        token_to_check = token.lower() if token in [CAP_TOKEN, ALLCAPS_TOKEN] else token 
        if token_to_check not in vocab:
            vocab[token_to_check] = next_id
            next_id += 1

    for token, count in token_counts.most_common():
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
            
    return vocab


def preprocess_example(example, vocab, numeric_feature_dim, max_seq_len):
    question = example['question']
    answer = example['answer']

    # Ensure padded_feat_row is a plain list of floats, correctly sized
    # This is critical for matching the schema's fixed length
    padded_feat_row = [0.0] * numeric_feature_dim 

    # --- Process Question (Encoder Input) ---
    encoder_tokens, encoder_nums, encoder_num_types = tokenize_and_featurize_text(
        question, NUM_TOKEN, PAD_TOKEN
    )
    encoder_token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in encoder_tokens]
    encoder_token_ids = [vocab[BOS_TOKEN]] + encoder_token_ids + [vocab[EOS_TOKEN]]
    
    # Ensure numeric_features are lists of lists of float32
    encoder_numeric_features = [
        # Explicitly convert to list and then to float for each element
        [float(val) for val in number_embedding_features(n_val, n_typ)]
        for n_val, n_typ in zip(encoder_nums, encoder_num_types)
    ]

    # Add padding for BOS/EOS tokens for numeric features
    encoder_numeric_features = [list(padded_feat_row)] + encoder_numeric_features + [list(padded_feat_row)]

    encoder_attention_mask = [True] * len(encoder_token_ids)

    # Pad/truncate encoder inputs
    encoder_token_ids, encoder_numeric_features, encoder_attention_mask = pad_sequence_for_model(
        encoder_token_ids, encoder_numeric_features, encoder_attention_mask,
        max_seq_len, vocab[PAD_TOKEN], padded_feat_row, is_encoder=True, vocab=vocab
    )

    # --- Process Answer (Decoder Input/Output) ---
    decoder_tokens, decoder_nums, decoder_num_types = tokenize_and_featurize_text(
        answer, NUM_TOKEN, PAD_TOKEN
    )
    
    # Decoder input: starts with BOS, then tokens from answer
    decoder_input_token_ids = [vocab[BOS_TOKEN]] + [vocab.get(token, vocab[UNK_TOKEN]) for token in decoder_tokens]
    
    decoder_input_numeric_features = [list(padded_feat_row)] + [
        [float(val) for val in number_embedding_features(n_val, n_typ)]
        for n_val, n_typ in zip(decoder_nums, decoder_num_types)
    ]

    # Decoder output targets: tokens from answer, then EOS
    decoder_output_token_targets = [vocab.get(token, vocab[UNK_TOKEN]) for token in decoder_tokens] + [vocab[EOS_TOKEN]]

    decoder_output_numeric_targets = [
        [float(val) for val in number_embedding_features(n_val, n_typ)]
        for n_val, n_typ in zip(decoder_nums, decoder_num_types)
    ] + [list(padded_feat_row)]

    decoder_attention_mask = [True] * len(decoder_input_token_ids)

    # Pad/truncate decoder inputs/outputs
    decoder_input_token_ids, decoder_input_numeric_features, decoder_attention_mask = pad_sequence_for_model(
        decoder_input_token_ids, decoder_input_numeric_features, decoder_attention_mask,
        max_seq_len, vocab[PAD_TOKEN], padded_feat_row, is_encoder=False, vocab=vocab
    )
    decoder_output_token_targets, decoder_output_numeric_targets, _ = pad_sequence_for_model(
        decoder_output_token_targets, decoder_output_numeric_targets, [True] * len(decoder_output_token_targets),
        max_seq_len, vocab[PAD_TOKEN], padded_feat_row, is_encoder=False, vocab=vocab
    )
    
    original_numbers_in_answer = [float(n) for n in re.findall(r'\d+\.?\d*', answer)]
    if not original_numbers_in_answer:
        original_numbers_in_answer = [0.0]

    return {
        'encoder_token_ids': encoder_token_ids,
        'encoder_numeric_features': encoder_numeric_features,
        'encoder_attention_mask': encoder_attention_mask,
        'decoder_input_token_ids': decoder_input_token_ids,
        'decoder_input_numeric_features': decoder_input_numeric_features,
        'decoder_output_token_targets': decoder_output_token_targets,
        'decoder_output_numeric_targets': decoder_output_numeric_targets,
        'decoder_attention_mask': decoder_attention_mask,
        'original_answers_text': answer,
        'original_numeric_values': original_numbers_in_answer
    }


def custom_collate_fn(batch, vocab, numeric_feature_dim, max_seq_len):
    encoder_token_ids_list = [item['encoder_token_ids'] for item in batch]
    encoder_numeric_features_list = [item['encoder_numeric_features'] for item in batch]
    encoder_attention_mask_list = [item['encoder_attention_mask'] for item in batch]
    decoder_input_token_ids_list = [item['decoder_input_token_ids'] for item in batch]
    decoder_input_numeric_features_list = [item['decoder_input_numeric_features'] for item in batch]
    decoder_output_token_targets_list = [item['decoder_output_token_targets'] for item in batch]
    decoder_output_numeric_targets_list = [item['decoder_output_numeric_targets'] for item in batch]
    decoder_attention_mask_list = [item['decoder_attention_mask'] for item in batch]
    
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

    return {
        'encoder_token_ids': padded_encoder_token_ids,
        'encoder_numeric_features': padded_encoder_numeric_features,
        'encoder_attention_mask': padded_encoder_attention_mask,
        'decoder_input_token_ids': padded_decoder_input_token_ids,
        'decoder_input_numeric_features': padded_decoder_input_numeric_features,
        'decoder_output_token_targets': padded_decoder_output_token_targets,
        'decoder_output_numeric_targets': padded_decoder_output_numeric_targets,
        'decoder_attention_mask': padded_decoder_attention_mask,
        'original_answers_text': original_answers_text_list,
        'original_numeric_values': original_numeric_values_list,
    }

def tokenize_and_featurize_text(text, num_token, pad_token):
    tokens = []
    numbers = []
    num_types = []
    
    parts = re.split(r'(\d+\.?\d*)', text)
    for part in parts:
        if re.fullmatch(r'\d+\.?\d*', part):
            tokens.append(num_token)
            numbers.append(float(part))
            num_types.append('float' if '.' in part else 'int')
        elif part:
            sub_tokens = re.findall(r"[\w']+|[.,!?;:()]|\s+", part.lower())
            for tok in sub_tokens:
                if tok.isspace():
                    if tokens and tokens[-1] != SPACE_TOKEN:
                        tokens.append(SPACE_TOKEN)
                else:
                    tokens.append(tok)
            non_space_tokens_count = len([t for t in sub_tokens if not t.isspace()])
            numbers.extend([0.0] * non_space_tokens_count)
            num_types.extend(['float'] * non_space_tokens_count)
    
    return tokens, numbers, num_types

def pad_sequence_for_model(token_ids, numeric_features, attention_mask, max_len, pad_id, padded_feat_row, is_encoder=True, vocab=None):
    if vocab is None:
        raise ValueError("vocab must be provided to pad_sequence_for_model")

    current_len = len(token_ids)
    
    # Ensure copies to prevent unexpected side effects
    token_ids_padded = list(token_ids)
    numeric_features_padded = [list(f) for f in numeric_features] # Deep copy for inner lists
    attention_mask_padded = list(attention_mask)

    if current_len > max_len:
        token_ids_padded = token_ids_padded[:max_len]
        numeric_features_padded = numeric_features_padded[:max_len]
        attention_mask_padded = attention_mask_padded[:max_len]
        
        if not is_encoder:
            if token_ids_padded and token_ids_padded[-1] != pad_id and token_ids_padded[-1] != vocab.get(EOS_TOKEN):
                token_ids_padded[-1] = vocab[EOS_TOKEN]
                # Ensure it's a new list of floats, matching numeric_feature_dim
                numeric_features_padded[-1] = [float(x) for x in padded_feat_row]
                attention_mask_padded[-1] = True
            elif not token_ids_padded and max_len > 0:
                token_ids_padded = [vocab[EOS_TOKEN]] + [pad_id] * (max_len - 1)
                numeric_features_padded = [[float(x) for x in padded_feat_row]] * max_len
                attention_mask_padded = [True] + [False] * (max_len - 1)

    elif current_len < max_len:
        padding_len = max_len - current_len
        token_ids_padded.extend([pad_id] * padding_len)
        # Each padded element must be a *copy* of padded_feat_row, not a reference
        numeric_features_padded.extend([[float(x) for x in padded_feat_row] for _ in range(padding_len)])
        attention_mask_padded.extend([False] * padding_len)

    # Crucial: verify lengths before returning
    if len(token_ids_padded) != max_len:
        raise ValueError(f"token_ids_padded length mismatch: expected {max_len}, got {len(token_ids_padded)}")
    if len(numeric_features_padded) != max_len:
        raise ValueError(f"numeric_features_padded length mismatch: expected {max_len}, got {len(numeric_features_padded)}")
    for i, features in enumerate(numeric_features_padded):
        if len(features) != len(padded_feat_row): # Should be numeric_feature_dim
            raise ValueError(f"Inner numeric feature vector length mismatch at index {i}: expected {len(padded_feat_row)}, got {len(features)}")
    if len(attention_mask_padded) != max_len:
        raise ValueError(f"attention_mask_padded length mismatch: expected {max_len}, got {len(attention_mask_padded)}")

    return token_ids_padded, numeric_features_padded, attention_mask_padded
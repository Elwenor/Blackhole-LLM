# File: data_processing.py

import re
from num2words import num2words
import torch
import numpy as np
from datasets import Features, Value, Sequence
import sys, os
from tqdm import tqdm
from collections import Counter

from config import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN, SPECIAL_TOKENS,
    CAP_TOKEN, ALLCAPS_TOKEN, SPACE_TOKEN, MAX_SEQ_LEN
)

# [POPRAWKA] Usunięto niepotrzebny import sys.path, zakładając standardową strukturę projektu.
from blackhole.embedding import number_embedding_features

def get_features_schema(numeric_feature_dim):
    """Defines the features schema for the dataset."""
    # [POPRAWKA] Usunięto 'length' z definicji Sequence, aby umożliwić dynamiczną długość,
    # co jest standardem w Hugging Face. Ostateczne dopełnienie i tak następuje w custom_collate_fn.
    return Features({
        'encoder_token_ids': Sequence(Value('int32')),
        'encoder_numeric_features': Sequence(Sequence(Value('float32'))),
        'encoder_attention_mask': Sequence(Value('bool')),
        'decoder_input_token_ids': Sequence(Value('int32')),
        'decoder_input_numeric_features': Sequence(Sequence(Value('float32'))),
        'decoder_output_token_targets': Sequence(Value('int32')),
        'decoder_output_numeric_targets': Sequence(Sequence(Value('float32'))),
        'decoder_attention_mask': Sequence(Value('bool')),
        'original_answers_text': Value('string'),
        'original_numeric_values': Sequence(Value('float32'))
    })

def basic_tokenize(text):
    """
    A simplified and robust tokenizer.
    Splits text into words, numbers (as NUM_TOKEN), and punctuation.
    Also handles spaces consistently.
    """
    # [POPRAWKA] Uproszczony i bardziej niezawodny regex.
    # Kolejność ma znaczenie: najpierw liczby, potem słowa, na końcu znaki.
    tokens = re.findall(r"(\d+\.?\d*)|[\w']+|[.,!?;:()]|\s+", text.lower())
    final_tokens = []
    for tok in tokens:
        if not tok:  # Pomiń puste stringi
            continue
        if tok.isspace():
            # Dodaj token spacji tylko jeśli poprzedni token nie był spacją
            if final_tokens and final_tokens[-1] != SPACE_TOKEN:
                final_tokens.append(SPACE_TOKEN)
        elif re.fullmatch(r'\d+\.?\d*', tok):
            final_tokens.append(NUM_TOKEN)
        else:
            final_tokens.append(tok)

    # Usuń wiodące/końcowe spacje
    if final_tokens and final_tokens[0] == SPACE_TOKEN:
        final_tokens.pop(0)
    if final_tokens and final_tokens[-1] == SPACE_TOKEN:
        final_tokens.pop()

    return final_tokens


def build_vocab_from_dataset(dataset_dict):
    """Builds a vocabulary from all splits of a dataset dictionary."""
    token_counts = Counter()
    print("Building vocabulary...")
    for split_name, current_dataset in dataset_dict.items():
        print(f"Processing split: {split_name}")
        for example in tqdm(current_dataset, desc=f"Building vocab from {split_name}"):
            question = example.get('question', '')
            answer = example.get('answer', '')
            if question:
                token_counts.update(basic_tokenize(question))
            if answer:
                token_counts.update(basic_tokenize(answer))

    # [POPRAWKA] Gwarantowana kolejność i obecność wszystkich specjalnych tokenów.
    vocab_list = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN, CAP_TOKEN, ALLCAPS_TOKEN, SPACE_TOKEN]
    vocab = {token: i for i, token in enumerate(vocab_list)}
    
    # Dodaj resztę tokenów posortowanych według częstości
    # Używamy `token.lower()` dla spójności, ponieważ `basic_tokenize` konwertuje na małe litery.
    for token, count in token_counts.most_common():
        token_lower = token.lower()
        if token_lower not in vocab:
            vocab[token_lower] = len(vocab)
            
    return vocab


def tokenize_and_extract_numbers(text):
    """
    Tokenizes text and extracts numerical values, keeping them aligned with tokens.
    """
    tokens = []
    numbers = []
    num_types = []
    # Regex, który znajduje liczby LUB inne części tekstu (słowa, interpunkcję)
    pattern = re.compile(r"(-?\d+\.?\d*)|([\w']+|[.,!?;:()])")
    
    # Używamy finditer, aby zachować kolejność i spacje
    last_idx = 0
    for match in pattern.finditer(text):
        # Obsługa spacji między tokenami
        if match.start() > last_idx:
            tokens.append(SPACE_TOKEN)
            numbers.append(float('nan'))
            num_types.append('none')
        
        number_str, other_tok = match.groups()
        
        if number_str:
            tokens.append(NUM_TOKEN)
            try:
                val = float(number_str)
                numbers.append(val)
                num_types.append('float' if '.' in number_str else 'int')
            except ValueError:
                numbers.append(float('nan'))
                num_types.append('none')
        elif other_tok:
            # Dzielimy słowa z wielkich liter
            if other_tok.isupper() and len(other_tok) > 1:
                tokens.append(ALLCAPS_TOKEN)
                tokens.append(other_tok.lower())
            elif other_tok[0].isupper() and len(other_tok) > 1:
                tokens.append(CAP_TOKEN)
                tokens.append(other_tok.lower())
            else:
                tokens.append(other_tok.lower())
            # Dodajemy tyle placeholderów, ile tokenów dodaliśmy
            for _ in range(tokens.count(other_tok.lower()) - (numbers.count(float('nan'))-tokens.count(SPACE_TOKEN))):
                numbers.append(float('nan'))
                num_types.append('none')

        last_idx = match.end()

    # Sprawdź, czy na końcu tekstu nie ma spacji
    if len(text) > last_idx:
        tokens.append(SPACE_TOKEN)
        numbers.append(float('nan'))
        num_types.append('none')

    # Zapewnienie spójności długości list
    while len(numbers) < len(tokens):
        numbers.append(float('nan'))
    while len(num_types) < len(tokens):
        num_types.append('none')

    return tokens, numbers, num_types


def preprocess_example(examples, vocab, numeric_feature_dim):
    """
    Preprocesses a batch of examples for the model.
    """
    batch_encoder_token_ids = []
    batch_encoder_numeric_features = []
    batch_encoder_attention_mask = []
    batch_decoder_input_token_ids = []
    batch_decoder_input_numeric_features = []
    batch_decoder_output_token_targets = []
    batch_decoder_output_numeric_targets = []
    batch_decoder_attention_mask = []
    batch_original_answers_text = []
    batch_original_numeric_values = []

    padded_feat_row = [-2.0] * numeric_feature_dim
    unk_token_id = vocab[UNK_TOKEN]
    bos_token_id = vocab[BOS_TOKEN]
    eos_token_id = vocab[EOS_TOKEN]

    for i in range(len(examples['question'])):
        question = examples['question'][i]
        answer = examples['answer'][i]

        # --- Process Question (Encoder Input) ---
        encoder_tokens, encoder_nums, encoder_num_types = tokenize_and_extract_numbers(question)
        encoder_token_ids = [bos_token_id] + [vocab.get(t, unk_token_id) for t in encoder_tokens] + [eos_token_id]
        
        encoder_numeric_features = [padded_feat_row] + \
            [number_embedding_features(val, typ, numeric_feature_dim).tolist() for val, typ in zip(encoder_nums, encoder_num_types)] + \
            [padded_feat_row]

        # --- Process Answer (Decoder Input/Output) ---
        decoder_tokens, decoder_nums, decoder_num_types = tokenize_and_extract_numbers(answer)
        
        decoder_input_token_ids = [bos_token_id] + [vocab.get(t, unk_token_id) for t in decoder_tokens]
        decoder_input_numeric_features = [padded_feat_row] + \
            [number_embedding_features(val, typ, numeric_feature_dim).tolist() for val, typ in zip(decoder_nums, decoder_num_types)]

        decoder_output_token_targets = [vocab.get(t, unk_token_id) for t in decoder_tokens] + [eos_token_id]
        decoder_output_numeric_targets = \
            [number_embedding_features(val, typ, numeric_feature_dim).tolist() for val, typ in zip(decoder_nums, decoder_num_types)] + \
            [padded_feat_row]

        # Append to batch lists (bez paddingu na tym etapie)
        batch_encoder_token_ids.append(encoder_token_ids)
        batch_encoder_numeric_features.append(encoder_numeric_features)
        batch_encoder_attention_mask.append([True] * len(encoder_token_ids))
        
        batch_decoder_input_token_ids.append(decoder_input_token_ids)
        batch_decoder_input_numeric_features.append(decoder_input_numeric_features)
        batch_decoder_output_token_targets.append(decoder_output_token_targets)
        batch_decoder_output_numeric_targets.append(decoder_output_numeric_targets)
        batch_decoder_attention_mask.append([True] * len(decoder_input_token_ids))
        
        # Store original data for evaluation
        batch_original_answers_text.append(answer)
        original_numbers = [n for n in decoder_nums if not np.isnan(n)]
        batch_original_numeric_values.append(original_numbers if original_numbers else [0.0])

    return {
        'encoder_token_ids': batch_encoder_token_ids,
        'encoder_numeric_features': batch_encoder_numeric_features,
        'encoder_attention_mask': batch_encoder_attention_mask,
        'decoder_input_token_ids': batch_decoder_input_token_ids,
        'decoder_input_numeric_features': batch_decoder_input_numeric_features,
        'decoder_output_token_targets': batch_decoder_output_token_targets,
        'decoder_output_numeric_targets': batch_decoder_output_numeric_targets,
        'decoder_attention_mask': batch_decoder_attention_mask,
        'original_answers_text': batch_original_answers_text,
        'original_numeric_values': batch_original_numeric_values
    }


def collate_and_pad_batch(batch, pad_token_id, numeric_feature_dim):
    """
    Custom collate function to pad sequences in a batch to the same length.
    """
    max_enc_len = max(len(item['encoder_token_ids']) for item in batch)
    max_dec_len = max(len(item['decoder_input_token_ids']) for item in batch)
    max_len = min(max(max_enc_len, max_dec_len), MAX_SEQ_LEN)
    
    padded_feat_row = [-2.0] * numeric_feature_dim

    for item in batch:
        # Pad Encoder
        enc_len = len(item['encoder_token_ids'])
        enc_rem = max_len - enc_len
        item['encoder_token_ids'] = (item['encoder_token_ids'] + [pad_token_id] * enc_rem)[:max_len]
        item['encoder_numeric_features'] = (item['encoder_numeric_features'] + [padded_feat_row] * enc_rem)[:max_len]
        item['encoder_attention_mask'] = (item['encoder_attention_mask'] + [False] * enc_rem)[:max_len]

        # Pad Decoder Input
        dec_in_len = len(item['decoder_input_token_ids'])
        dec_in_rem = max_len - dec_in_len
        item['decoder_input_token_ids'] = (item['decoder_input_token_ids'] + [pad_token_id] * dec_in_rem)[:max_len]
        item['decoder_input_numeric_features'] = (item['decoder_input_numeric_features'] + [padded_feat_row] * dec_in_rem)[:max_len]
        item['decoder_attention_mask'] = (item['decoder_attention_mask'] + [False] * dec_in_rem)[:max_len]

        # Pad Decoder Output
        dec_out_len = len(item['decoder_output_token_targets'])
        dec_out_rem = max_len - dec_out_len
        item['decoder_output_token_targets'] = (item['decoder_output_token_targets'] + [pad_token_id] * dec_out_rem)[:max_len]
        item['decoder_output_numeric_targets'] = (item['decoder_output_numeric_targets'] + [padded_feat_row] * dec_out_rem)[:max_len]

    # Convert lists to tensors
    return {
        'encoder_token_ids': torch.tensor([item['encoder_token_ids'] for item in batch], dtype=torch.long),
        'encoder_numeric_features': torch.tensor([item['encoder_numeric_features'] for item in batch], dtype=torch.float32),
        'encoder_attention_mask': torch.tensor([item['encoder_attention_mask'] for item in batch], dtype=torch.bool),
        'decoder_input_token_ids': torch.tensor([item['decoder_input_token_ids'] for item in batch], dtype=torch.long),
        'decoder_input_numeric_features': torch.tensor([item['decoder_input_numeric_features'] for item in batch], dtype=torch.float32),
        'decoder_output_token_targets': torch.tensor([item['decoder_output_token_targets'] for item in batch], dtype=torch.long),
        'decoder_output_numeric_targets': torch.tensor([item['decoder_output_numeric_targets'] for item in batch], dtype=torch.float32),
        'decoder_attention_mask': torch.tensor([item['decoder_attention_mask'] for item in batch], dtype=torch.bool),
        'original_answers_text': [item['original_answers_text'] for item in batch],
        'original_numeric_values': [item['original_numeric_values'] for item in batch],
    }
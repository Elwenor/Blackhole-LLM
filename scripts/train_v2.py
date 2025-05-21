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

# Hugging Face datasets and transformers
from datasets import load_dataset, Dataset, Features, Value, Sequence
from transformers import get_scheduler

# Safe __file__ usage with proper error handling
try:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, base_dir)

# Set seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Import from your blackhole modules
from blackhole.tokenizer import tokenize
from blackhole.embedding import TokenEmbedding, NumberEmbedding, number_embedding_features, decode_number_from_features
import blackhole.embedding

# Import training and evaluation functions from nova.py
from blackhole.nova import train_step, evaluate, focal_loss, mse_loss_for_numerical_features

import matplotlib.pyplot as plt

print(f"DEBUG: Załadowano embedding.py z: {blackhole.embedding.__file__}")


# --- DEFINITION OF THE ENCODER-DECODER MODEL ---
class ImprovedCrossEmbeddingSeq2SeqModel(nn.Module):
    """
    Model Encoder-Decoder łączący osadzenia tokenów i liczb za pomocą Transformerów.
    Zaprojektowany do zadań Question Answering z odpowiedziami zawierającymi tekst i liczby.
    """
    def __init__(self, vocab_size, token_dim=128, num_dim=128, hidden=256,
                 encoder_layers=3, decoder_layers=3, dropout=0.2, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden = hidden

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=token_dim)
        self.num_embedding = NumberEmbedding(input_dim=feature_dim, output_dim=num_dim)

        # Projections to common hidden dimension
        self.token_to_common_dim = nn.Linear(token_dim, hidden)
        self.num_to_common_dim = nn.Linear(num_dim, hidden)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Output Heads
        self.token_head = nn.Linear(hidden, vocab_size)
        self.num_head = nn.Sequential(
            nn.Linear(hidden, feature_dim),
            nn.Tanh()
        )

    def forward(self, encoder_token_ids, encoder_numeric_features, encoder_attention_mask,
                decoder_token_ids, decoder_numeric_features_input, decoder_attention_mask):

        enc_token_emb = self.token_embedding(encoder_token_ids)
        enc_num_emb = self.num_embedding(encoder_numeric_features)

        enc_token_emb_proj = self.token_to_common_dim(enc_token_emb)
        enc_num_emb_proj = self.num_to_common_dim(enc_num_emb)

        encoder_input_emb = enc_token_emb_proj + enc_num_emb_proj

        encoder_output = self.transformer_encoder(encoder_input_emb, src_key_padding_mask=encoder_attention_mask)

        dec_token_emb = self.token_embedding(decoder_token_ids)
        dec_num_emb = self.num_embedding(decoder_numeric_features_input)

        dec_token_emb_proj = self.token_to_common_dim(dec_token_emb)
        dec_num_emb_proj = self.num_to_common_dim(dec_num_emb)

        decoder_input_emb = dec_token_emb_proj + dec_num_emb_proj

        tgt_seq_len = decoder_token_ids.size(1)
        # Tworzymy maskę, która maskuje przyszłe tokeny (auto-regresja)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(decoder_token_ids.device)

        decoder_output = self.transformer_decoder(
            tgt=decoder_input_emb,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_attention_mask,
            memory_key_padding_mask=encoder_attention_mask
        )

        token_logits = self.token_head(decoder_output)
        num_feature_output = self.num_head(decoder_output)

        return token_logits, num_feature_output


# --- DATA PREPARATION ---
def preprocess_example(example, vocab, feature_embedding_dim, max_seq_len=128):
    """
    Preprocesses a single example for the Encoder-Decoder model.
    Handles tokenization of both input question and target answer,
    replacing numbers in the answer with <|num|> and storing their features.
    Returns lists of lists for numeric features, to be converted to tensors in collate_fn.

    Args:
        example (dict): A dictionary with 'question' and 'answer' keys.
        vocab (dict): Token to index mapping.
        feature_embedding_dim (int): Dimension of numerical features.
        max_seq_len (int): Maximum sequence length for padding/truncation.

    Returns:
        dict: Processed data as Python lists (including lists of lists for numeric features).
    """
    pad_token_id = vocab.get('<|pad|>', 0)
    unk_token_id = vocab.get('<|unk|>', 0)
    num_token_id = vocab.get('<|num|>', unk_token_id)
    bos_token_id = vocab.get('<|bos|>', unk_token_id)
    eos_token_id = vocab.get('<|eos|>', unk_token_id)

    # padded_feat_row should be a list of floats, not a tensor, since we want lists of lists
    padded_feat_row_list = [-2.0] * feature_embedding_dim

    # --- Encoder Input Processing (Question) ---
    enc_tokens, enc_number_map = tokenize(example['question'])
    enc_token_ids = [vocab.get(tok, unk_token_id) for tok in enc_tokens]
    
    _enc_numeric_features = [] # List of lists of floats
    for token_idx in range(len(enc_token_ids)):
        if token_idx in enc_number_map:
            val, typ, raw = enc_number_map[token_idx]
            # Convert the returned tensor from number_embedding_features to a list
            _enc_numeric_features.append(number_embedding_features(val, typ, dim=feature_embedding_dim).tolist())
        else:
            _enc_numeric_features.append(padded_feat_row_list)

    # --- Decoder Target Processing (Answer) ---
    ans_tokens, ans_number_map = tokenize(example['answer'])

    _dec_target_token_ids = [bos_token_id]
    _dec_target_numeric_features = [padded_feat_row_list] # List of lists of floats
    _target_numerical_values_value = [float('nan')] # Padding for BOS
    _target_numerical_values_type = ["padding"] # Padding for BOS
    _target_numerical_values_raw_string = ["<pad_num>"] # Padding for BOS


    for i, tok in enumerate(ans_tokens):
        if i in ans_number_map:
            val, typ, raw = ans_number_map[i]
            _dec_target_token_ids.append(num_token_id)
            # Convert the returned tensor from number_embedding_features to a list
            _dec_target_numeric_features.append(number_embedding_features(val, typ, dim=feature_embedding_dim).tolist())
            _target_numerical_values_value.append(float(val))
            _target_numerical_values_type.append(str(typ))
            _target_numerical_values_raw_string.append(str(raw))
        else:
            _dec_target_token_ids.append(vocab.get(tok, unk_token_id))
            _dec_target_numeric_features.append(padded_feat_row_list)
            _target_numerical_values_value.append(float('nan')) # Padding for text tokens
            _target_numerical_values_type.append("padding")
            _target_numerical_values_raw_string.append("<pad_num>")

    _dec_target_token_ids.append(eos_token_id)
    _dec_target_numeric_features.append(padded_feat_row_list)
    _target_numerical_values_value.append(float('nan')) # Padding for EOS
    _target_numerical_values_type.append("padding")
    _target_numerical_values_raw_string.append("<pad_num>")

    # Perform slicing for decoder input and output based on _dec_target_token_ids
    decoder_input_token_ids = _dec_target_token_ids[:-1]
    decoder_input_numeric_features = _dec_target_numeric_features[:-1] # This is a list of lists

    decoder_output_token_targets = _dec_target_token_ids[1:]
    decoder_output_numeric_targets = _dec_target_numeric_features[1:] # This is a list of lists

    # Apply the same slicing to the numerical target lists
    target_numerical_values_value = _target_numerical_values_value[1:]
    target_numerical_values_type = _target_numerical_values_type[1:]
    target_numerical_values_raw_string = _target_numerical_values_raw_string[1:]


    # --- Padding and Truncation ---
    # Encoder
    enc_len = len(enc_token_ids)
    enc_attention_mask = [False] * enc_len
    if enc_len > max_seq_len:
        enc_token_ids = enc_token_ids[:max_seq_len]
        _enc_numeric_features = _enc_numeric_features[:max_seq_len]
        enc_attention_mask = enc_attention_mask[:max_seq_len]
    else:
        num_padding_needed_enc = max_seq_len - enc_len
        enc_token_ids += [pad_token_id] * num_padding_needed_enc
        _enc_numeric_features += [padded_feat_row_list] * num_padding_needed_enc
        enc_attention_mask += [True] * num_padding_needed_enc


    # Decoder
    dec_len_after_slicing = len(decoder_input_token_ids)
    dec_attention_mask = [False] * dec_len_after_slicing

    if dec_len_after_slicing > max_seq_len:
        decoder_input_token_ids = decoder_input_token_ids[:max_seq_len]
        decoder_input_numeric_features = decoder_input_numeric_features[:max_seq_len]
        decoder_output_token_targets = decoder_output_token_targets[:max_seq_len]
        decoder_output_numeric_targets = decoder_output_numeric_targets[:max_seq_len]
        dec_attention_mask = dec_attention_mask[:max_seq_len]
        target_numerical_values_value = target_numerical_values_value[:max_seq_len]
        target_numerical_values_type = target_numerical_values_type[:max_seq_len]
        target_numerical_values_raw_string = target_numerical_values_raw_string[:max_seq_len]
    else:
        num_padding_needed_dec = max_seq_len - dec_len_after_slicing
        decoder_input_token_ids += [pad_token_id] * num_padding_needed_dec
        decoder_input_numeric_features += [padded_feat_row_list] * num_padding_needed_dec
        decoder_output_token_targets += [pad_token_id] * num_padding_needed_dec
        decoder_output_numeric_targets += [padded_feat_row_list] * num_padding_needed_dec
        dec_attention_mask += [True] * num_padding_needed_dec

        target_numerical_values_value.extend([float('nan')] * num_padding_needed_dec)
        target_numerical_values_type.extend(["padding"] * num_padding_needed_dec)
        target_numerical_values_raw_string.extend(["<pad_num>"] * num_padding_needed_dec)

    # Assertions to ensure correct lengths
    assert len(enc_token_ids) == max_seq_len, f"Encoder tokens length: {len(enc_token_ids)}, expected: {max_seq_len}"
    assert len(_enc_numeric_features) == max_seq_len, f"Encoder numeric features length: {len(_enc_numeric_features)}, expected: {max_seq_len}"
    assert all(len(f) == feature_embedding_dim for f in _enc_numeric_features), "Encoder numeric features inner list has wrong dimension"
    assert len(enc_attention_mask) == max_seq_len, f"Encoder attention mask length: {len(enc_attention_mask)}, expected: {max_seq_len}"

    assert len(decoder_input_token_ids) == max_seq_len, f"Decoder input tokens length: {len(decoder_input_token_ids)}, expected: {max_seq_len}"
    assert len(decoder_input_numeric_features) == max_seq_len, f"Decoder input numeric features length: {len(decoder_input_numeric_features)}, expected: {max_seq_len}"
    assert all(len(f) == feature_embedding_dim for f in decoder_input_numeric_features), "Decoder input numeric features inner list has wrong dimension"
    assert len(decoder_output_token_targets) == max_seq_len, f"Decoder output token targets length: {len(decoder_output_token_targets)}, expected: {max_seq_len}"
    assert len(decoder_output_numeric_targets) == max_seq_len, f"Decoder output numeric targets length: {len(decoder_output_numeric_targets)}, expected: {max_seq_len}"
    assert all(len(f) == feature_embedding_dim for f in decoder_output_numeric_targets), "Decoder output numeric targets inner list has wrong dimension"
    assert len(dec_attention_mask) == max_seq_len, f"Decoder attention mask length: {len(dec_attention_mask)}, expected: {max_seq_len}"

    assert len(target_numerical_values_value) == max_seq_len, f"Numerical values (value) length: {len(target_numerical_values_value)}, expected: {max_seq_len}"
    assert len(target_numerical_values_type) == max_seq_len, f"Numerical values (type) length: {len(target_numerical_values_type)}, expected: {max_seq_len}"
    assert len(target_numerical_values_raw_string) == max_seq_len, f"Numerical values (raw_string) length: {len(target_numerical_values_raw_string)}, expected: {max_seq_len}"


    return {
        'encoder_token_ids': enc_token_ids, # List of ints
        'encoder_numeric_features': _enc_numeric_features, # Now explicitly list of lists of floats
        'encoder_attention_mask': enc_attention_mask, # List of bools
        'decoder_input_token_ids': decoder_input_token_ids, # List of ints
        'decoder_input_numeric_features': decoder_input_numeric_features, # Now explicitly list of lists of floats
        'decoder_output_token_targets': decoder_output_token_targets, # List of ints
        'decoder_output_numeric_targets': decoder_output_numeric_targets, # Now explicitly list of lists of floats
        'decoder_attention_mask': dec_attention_mask, # List of bools
        'target_numerical_values_value': target_numerical_values_value, # List of floats
        'target_numerical_values_type': target_numerical_values_type, # List of strings
        'target_numerical_values_raw_string': target_numerical_values_raw_string, # List of strings
        'original_question': example['question'],
        'original_answer': example['answer']
    }

def custom_collate_fn(batch_examples, vocab, feature_embedding_dim, max_seq_len=128):
    """
    Collate function to stack processed examples into a batch.
    Converts lists from preprocess_example into tensors for batching.
    """
    enc_token_ids_batch = []
    enc_numeric_features_batch = []
    enc_attention_mask_batch = []
    dec_input_token_ids_batch = []
    dec_input_numeric_features_batch = []
    dec_output_token_targets_batch = []
    dec_output_numeric_targets_batch = []
    dec_attention_mask_batch = []
    original_question_strings_batch = []
    original_answer_strings_batch = []

    target_numerical_values_value_batch = []
    target_numerical_values_type_batch = []
    target_numerical_values_raw_string_batch = []


    for ex in batch_examples:
        # Convert lists to tensors and append for stacking
        enc_token_ids_batch.append(torch.tensor(ex['encoder_token_ids'], dtype=torch.long))
        # Now we expect ex['encoder_numeric_features'] to be a list of lists of floats
        enc_numeric_features_batch.append(torch.tensor(ex['encoder_numeric_features'], dtype=torch.float32))
        enc_attention_mask_batch.append(torch.tensor(ex['encoder_attention_mask'], dtype=torch.bool))

        dec_input_token_ids_batch.append(torch.tensor(ex['decoder_input_token_ids'], dtype=torch.long))
        dec_input_numeric_features_batch.append(torch.tensor(ex['decoder_input_numeric_features'], dtype=torch.float32))
        dec_output_token_targets_batch.append(torch.tensor(ex['decoder_output_token_targets'], dtype=torch.long))
        dec_output_numeric_targets_batch.append(torch.tensor(ex['decoder_output_numeric_targets'], dtype=torch.float32))
        dec_attention_mask_batch.append(torch.tensor(ex['decoder_attention_mask'], dtype=torch.bool))

        target_numerical_values_value_batch.append(ex['target_numerical_values_value'])
        target_numerical_values_type_batch.append(ex['target_numerical_values_type'])
        target_numerical_values_raw_string_batch.append(ex['target_numerical_values_raw_string'])

        original_question_strings_batch.append(ex['original_question'])
        original_answer_strings_batch.append(ex['original_answer'])

    return {
        'encoder_token_ids': torch.stack(enc_token_ids_batch),
        'encoder_numeric_features': torch.stack(enc_numeric_features_batch),
        'encoder_attention_mask': torch.stack(enc_attention_mask_batch),
        'decoder_input_token_ids': torch.stack(dec_input_token_ids_batch),
        'decoder_input_numeric_features': torch.stack(dec_input_numeric_features_batch),
        'decoder_output_token_targets': torch.stack(dec_output_token_targets_batch),
        'decoder_output_numeric_targets': torch.stack(dec_output_numeric_targets_batch),
        'decoder_attention_mask': torch.stack(dec_attention_mask_batch),
        'target_numerical_values_value': target_numerical_values_value_batch,
        'target_numerical_values_type': target_numerical_values_type_batch,
        'target_numerical_values_raw_string': target_numerical_values_raw_string_batch,
        'original_question_strings': original_question_strings_batch,
        'original_answer_strings': original_answer_strings_batch
    }

def build_vocab_from_dataset(dataset):
    """
    Builds vocabulary from a Hugging Face dataset.
    Assumes dataset structure with 'question' and 'answer' keys.
    """
    all_tokens = set()
    special_tokens = {'<|pad|>', '<|unk|>', '<|bos|>', '<|eos|>', '<|cap|>', '<|allcaps|>', '<|num|>', '<|space|>'}
    all_tokens.update(special_tokens)

    # GSM8K has 'train' and 'test' splits
    for split_name in ['train', 'test']:
        if split_name in dataset:
            for example in tqdm(dataset[split_name], desc=f"Building vocab from {split_name} split"):
                q_tokens, _ = tokenize(example['question'])
                all_tokens.update(q_tokens)

                a_tokens, _ = tokenize(example['answer'])
                all_tokens.update(a_tokens)

    vocab = {tok: i for i, tok in enumerate(sorted(list(all_tokens)))}
    return vocab

# Funkcja do definiowania schematu danych dla PyArrow
def get_features_schema(feature_embedding_dim, max_seq_len):
    # Features schema should reflect what preprocess_example RETURNS,
    # which are Python lists of lists for numeric features.
    return Features({
        "encoder_token_ids": Sequence(Value("int64"), length=max_seq_len),
        # This will be a list of lists of floats from preprocess_example
        "encoder_numeric_features": Sequence(Sequence(Value("float32"), length=feature_embedding_dim), length=max_seq_len),
        "encoder_attention_mask": Sequence(Value("bool"), length=max_seq_len),
        "decoder_input_token_ids": Sequence(Value("int64"), length=max_seq_len),
        "decoder_input_numeric_features": Sequence(Sequence(Value("float32"), length=feature_embedding_dim), length=max_seq_len),
        "decoder_output_token_targets": Sequence(Value("int64"), length=max_seq_len),
        "decoder_output_numeric_targets": Sequence(Sequence(Value("float32"), length=feature_embedding_dim), length=max_seq_len),
        "decoder_attention_mask": Sequence(Value("bool"), length=max_seq_len),
        "target_numerical_values_value": Sequence(Value("float64"), length=max_seq_len),
        "target_numerical_values_type": Sequence(Value("string"), length=max_seq_len),
        "target_numerical_values_raw_string": Sequence(Value("string"), length=max_seq_len),
        "original_question": Value("string"),
        "original_answer": Value("string")
    })

def predict_and_decode_answer(model, encoder_token_ids, encoder_numeric_features, encoder_attention_mask, vocab, device, max_decoding_len=128, padded_feat_row=None):
    model.eval()
    idx_to_token = {idx: token for token, idx in vocab.items()}
    num_token_id = vocab.get('<|num|>', vocab.get('<|unk|>', 0))
    bos_token_id = vocab.get('<|bos|>', vocab.get('<|unk|>', 0))
    eos_token_id = vocab.get('<|eos|>', vocab.get('<|unk|>', 0))
    pad_token_id = vocab.get('<|pad|>', 0)
    
    if padded_feat_row is None:
        if hasattr(model, 'feature_dim'):
            padded_feat_row = torch.full((model.feature_dim,), -2.0, dtype=torch.float32).to(device)
        else:
            raise ValueError("Model must have a 'feature_dim' attribute or 'padded_feat_row' must be explicitly provided.")

    batch_size = encoder_token_ids.size(0)

    decoder_input_token_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    decoder_input_numeric_features = padded_feat_row.unsqueeze(0).repeat(batch_size, 1, 1)
    decoder_attention_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)

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

            for i in range(batch_size):
                if next_token_ids[i].item() == eos_token_id:
                    if len(generated_tokens_list[i]) == 0 or generated_tokens_list[i][-1] != 'STOP_DECODING':
                        generated_tokens_list[i].append('STOP_DECODING')
                elif generated_tokens_list[i] and generated_tokens_list[i][-1] == 'STOP_DECODING':
                    continue
                else:
                    token = idx_to_token.get(next_token_ids[i].item(), '<|unk|>')
                    generated_tokens_list[i].append(token)
                    if token == '<|num|>':
                        decoded_val = decode_number_from_features(next_num_features[i].cpu().numpy())
                        generated_num_values_list[i].append(decoded_val)
                    else:
                        generated_num_values_list[i].append(None)

            if all('STOP_DECODING' in seq for seq in generated_tokens_list):
                break

    final_decoded_answers = []
    for i in range(batch_size):
        predicted_answer_tokens = []
        current_generated_num_idx = 0
        temp_generated_tokens = [tok for tok in generated_tokens_list[i] if tok != 'STOP_DECODING']

        k = 0
        while k < len(temp_generated_tokens):
            token = temp_generated_tokens[k]
            if token == '<|num|>':
                val = generated_num_values_list[i][current_generated_num_idx]
                if val is not None:
                    if abs(val - round(val)) < 1e-6:
                        predicted_answer_tokens.append(str(int(round(val))))
                    else:
                        predicted_answer_tokens.append(f"{val:.2f}")
                current_generated_num_idx += 1
            elif token == '<|cap|>':
                if k + 1 < len(temp_generated_tokens) and temp_generated_tokens[k+1] not in ['<|cap|>', '<|allcaps|>', '<|num|>', '<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|space|>']:
                    predicted_answer_tokens.append(temp_generated_tokens[k+1].capitalize())
                    k += 1
            elif token == '<|allcaps|>':
                if k + 1 < len(temp_generated_tokens) and temp_generated_tokens[k+1] not in ['<|cap|>', '<|allcaps|>', '<|num|>', '<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|space|>']:
                    predicted_answer_tokens.append(temp_generated_tokens[k+1].upper())
                    k += 1
            elif token == '<|space|>':
                predicted_answer_tokens.append(' ')
            elif token == '<|pad|>':
                pass
            else:
                predicted_answer_tokens.append(token)
            k += 1
        
        predicted_answer_raw = "".join(predicted_answer_tokens)
        predicted_answer_cleaned = re.sub(r'\s([.,!?;:])', r'\1', predicted_answer_raw)
        predicted_answer_cleaned = re.sub(r'\s+', ' ', predicted_answer_cleaned).strip()
        final_decoded_answers.append(predicted_answer_cleaned)

    return final_decoded_answers

# Example for tokenize (used for displaying input)
def tokenize(token_ids, vocab):
    idx_to_token = {idx: token for token, idx in vocab.items()}
    return ' '.join([idx_to_token.get(idx, '<|unk|>') for idx in token_ids if idx != vocab.get('<|pad|>')])


# --- Your main execution block ---
if __name__ == '__main__':
    device = torch.device("cpu") # Change to "cuda" if GPU is available and set up
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")


    # --- KONFIGURACJA DANYCH I WYŚWIETLANIA ---
    # Ustaw na 1.0 (100%) dla pełnego treningu
    # Ustaw na np. 0.1 (10%) dla szybkiego testu
    data_sample_percentage = 0.1 # <--- ZMIEN TUTAJ, ABY KONTROLOWAĆ ROZMIAR DATASETU
    
    # Procent datasetu walidacyjnego używany do PEŁNEJ EWALUACJI (z dekodowaniem).
    # Ustaw na małą wartość (np. 0.05 lub 0.01) aby przyspieszyć evaluate().
    eval_sample_percentage = 0.05 # <--- NOWA ZMIENNA DLA KONTROLI SZYBKOSCI EWALUACJI

    # Liczba przykładów do wyświetlenia po każdej epoce. Ustawiono na 1 dla debugowania.
    num_examples_to_display = 1
    
    # Load dataset
    print("Loading dataset...")
    try:
        ds = load_dataset("openai/gsm8k", "main")
        train_dataset_full = ds['train']
        val_dataset_full = ds['test']
        print(f"Loaded full dataset: {len(train_dataset_full)} training examples, {len(val_dataset_full)} validation examples.")

        # Próbkowanie datasetu, jeśli data_sample_percentage < 1.0
        if data_sample_percentage < 1.0:
            train_size = int(len(train_dataset_full) * data_sample_percentage)
            val_size = int(len(val_dataset_full) * data_sample_percentage)
            
            # Wylosuj indeksy do próbkowania
            train_indices = random.sample(range(len(train_dataset_full)), train_size)
            val_indices_full_sample = random.sample(range(len(val_dataset_full)), val_size)

            train_dataset = train_dataset_full.select(train_indices)
            val_dataset_for_full_eval = val_dataset_full.select(val_indices_full_sample) # Used for full eval.
            print(f"Using sampled dataset: {len(train_dataset)} training examples, {len(val_dataset_for_full_eval)} validation examples for full eval ({data_sample_percentage*100:.0f}% of full).")
        else:
            train_dataset = train_dataset_full
            val_dataset_for_full_eval = val_dataset_full # Use full for full eval.
            print("Using full dataset for training and full evaluation.")

        # Próbkowanie dla SZYBKIEJ EWALUACJI (tylko do podglądu lub jeśli evaluate jest bardzo wolne)
        if eval_sample_percentage < 1.0:
            eval_size = max(1, int(len(val_dataset_for_full_eval) * eval_sample_percentage))
            eval_indices = random.sample(range(len(val_dataset_for_full_eval)), eval_size)
            val_dataset_for_evaluation = val_dataset_for_full_eval.select(eval_indices)
            print(f"Using sampled validation dataset for actual evaluation: {len(val_dataset_for_evaluation)} examples ({eval_sample_percentage*100:.0f}% of sampled validation).")
        else:
            val_dataset_for_evaluation = val_dataset_for_full_eval
            print("Using full sampled validation dataset for actual evaluation.")


    except Exception as e:
        print(f"Failed to load 'openai/gsm8k' dataset: {e}")
        print("Creating dummy dataset for demonstration purposes as a fallback.")
        dummy_data = {
            'question': [
                "What is 5 plus 3?", "Subtract 10 from 20.",
                "How many apples are there if you have 7 and get 2 more?",
                "What is the result of 15 minus 8?",
                "The temperature was 25 degrees Celsius and dropped by 5 degrees. What is the new temperature?",
                "If you have fifty-five and add twenty-three, what do you get?",
                "One hundred minus forty-two is what?",
                "Three times four is?", "What is half of 10?",
                "If you have 100 and spend 25, what's left?",
                "What is 10 plus 20 minus 5?", "How much is 10 times 2?",
                "The number 7 plus the number 8 equals?",
                "What is 12 divided by 3?", "Two hundred plus fifty is what?",
                "What is 9 minus 4?", "Add 1 to 10.", "Double 6.",
                "Half of 14.", "How much is 11 plus 11?",
            ],
            'answer': [
                "The answer is 8.", "It's 10.", "You have 9 apples.",
                "The result is 7.", "The new temperature is 20 degrees Celsius.",
                "You get 78.", "Fifty-eight.", "Twelve.", "Five.",
                "75 is left.", "25.", "20.", "15.", "4.",
                "Two hundred fifty.", "5.", "11.", "12.", "7.", "22.",
            ]
        }
        train_dataset = Dataset.from_dict(dummy_data)
        val_dataset_for_full_eval = Dataset.from_dict(dummy_data) # Renamed for clarity
        val_dataset_for_evaluation = Dataset.from_dict(dummy_data) # Used for evaluate() call
        ds = {'train': train_dataset, 'test': val_dataset_for_full_eval}


    print("Building vocabulary...")
    vocab = build_vocab_from_dataset(ds)
    vocab_size = len(vocab)
    idx_to_token = {idx: token for token, idx in vocab.items()} # Potrzebne do dekodowania
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens in vocab: <|num|> = {vocab.get('<|num|>')}, <|bos|> = {vocab.get('<|bos|>')}, <|eos|> = {vocab.get('<|eos|>')}, <|pad|> = {vocab.get('<|pad|>')}")

    sample_val = 123.0
    sample_type = 'int'
    determined_numeric_feature_dim = len(number_embedding_features(sample_val, sample_type))
    print(f"Determined numeric feature dimension: {determined_numeric_feature_dim}")

    MAX_SEQ_LEN = 128
    features_schema = get_features_schema(determined_numeric_feature_dim, MAX_SEQ_LEN)

    print("Initializing model...")
    model = ImprovedCrossEmbeddingSeq2SeqModel(
        vocab_size=vocab_size,
        token_dim=128,
        num_dim=128,
        hidden=256,
        encoder_layers=3,
        decoder_layers=3,
        dropout=0.1,
        feature_dim=determined_numeric_feature_dim
    ).to(device)

    print(f"Model initialized with vocab size: {vocab_size}, numeric feature input dim: {determined_numeric_feature_dim}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 20
    batch_size = 32
    gradient_accumulation_steps = 1

    num_training_steps = max(1, (len(train_dataset) // batch_size // gradient_accumulation_steps) * num_epochs)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_val_numerical_accuracy = -1.0
    train_ce_losses = []
    train_num_losses = []
    train_total_losses = []
    val_exact_accuracies = []
    val_numerical_accuracies = []

    # --- WAŻNA POPRAWKA: ZDEFINIOWANIE padded_feat_row GLOBALNIE ---
    # Zrób to RAZ, po zainicjowaniu modelu i urządzenia.
    global_padded_feat_row = torch.full((model.feature_dim,), -2.0, dtype=torch.float32).to(device)


    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_ce_loss = 0.0
        total_numeric_loss = 0.0
        total_overall_loss = 0.0
        num_batches = 0

        # Wartość load_from_cache_file ustawiona na False w celu debugowania, zmień na True, gdy jesteś pewien, że preprocessing jest stabilny
        processed_train_dataset = train_dataset.map(
            lambda x: preprocess_example(x, vocab, determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN),
            remove_columns=['question', 'answer'],
            load_from_cache_file=True, # Zmienione na True
            desc="Preprocessing training examples",
            features=features_schema
        )

        train_dataloader = torch.utils.data.DataLoader(
            processed_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: custom_collate_fn(b, vocab, determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN)
        )

        optimizer.zero_grad()

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for i, batch in enumerate(train_dataloader):
                stats = train_step(model, optimizer, scheduler, batch, vocab, device, clip_grad=1.0)

                total_ce_loss += stats['ce_loss']
                total_numeric_loss += stats['numeric_loss']
                total_overall_loss += stats['total_loss']
                num_batches += 1

                pbar.set_postfix({
                    'CE': f"{stats['ce_loss']:.4f}",
                    'NumL': f"{stats['numeric_loss']:.4f}",
                    'TotalL': f"{stats['total_loss']:.4f}"
                })
                pbar.update(1)

        avg_ce_loss = total_ce_loss / num_batches
        avg_numeric_loss = total_numeric_loss / num_batches
        avg_total_loss = total_overall_loss / num_batches

        train_ce_losses.append(avg_ce_loss)
        train_num_losses.append(avg_numeric_loss)
        train_total_losses.append(avg_total_loss)

        print(f"\nEpoch {epoch+1} Avg Stats: CE: {avg_ce_loss:.4f}, Num Loss: {avg_numeric_loss:.4f}, Total Loss: {avg_total_loss:.4f}")

        # Evaluation
        print("Evaluating on validation set...")
        # Only preprocess the smaller sample for evaluation
        processed_val_dataset_for_evaluation = val_dataset_for_evaluation.map(
            lambda x: preprocess_example(x, vocab, determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN),
            remove_columns=['question', 'answer'],
            load_from_cache_file=True,
            desc="Preprocessing validation examples for evaluation",
            features=features_schema
        )

        val_exact_accuracy, val_numerical_accuracy = evaluate(
            model,
            processed_val_dataset_for_evaluation, # Użyj mniejszego datasetu tutaj
            vocab,
            device,
            batch_size=batch_size,
            collate_fn=lambda b: custom_collate_fn(b, vocab, determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN)
        )
        val_exact_accuracies.append(val_exact_accuracy)
        val_numerical_accuracies.append(val_numerical_accuracy)

        print(f"Validation Exact Match Accuracy: {val_exact_accuracy:.4f}")
        print(f"Validation Numerical Accuracy: {val_numerical_accuracy:.4f}")

        if val_numerical_accuracy > best_val_numerical_accuracy:
            best_val_numerical_accuracy = val_numerical_accuracy
            torch.save(model.state_dict(), "best_model_v3.pth")
            print(f"New best model saved with numerical accuracy: {best_val_numerical_accuracy:.4f}")

        # --- Podgląd predykcji po każdej epoce ---
        print("\n--- Model Predictions Sample ---")
        model.eval()
        # Wybieramy jeden losowy przykład z walidacyjnego datasetu do wyświetlenia
        sample_index = random.choice(range(len(val_dataset_for_full_eval)))
        example = val_dataset_for_full_eval[sample_index] # Pobierz oryginalny przykład

        # --- WAŻNA POPRAWKA: PRZYGOTOWANIE POJEDYNCZEGO PRZYKŁADU DO PREDYKCJI ---
        # Musimy przetworzyć przykład do formatu tensorów, który przyjmuje model
        processed_example_for_prediction = preprocess_example(example, vocab, determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN)
        
        # Użyj custom_collate_fn dla pojedynczego przykładu, aby uzyskać odpowiednie tensory.
        # Musimy owinąć go w listę, ponieważ custom_collate_fn oczekuje listy.
        collated_sample_for_prediction = custom_collate_fn(
            [processed_example_for_prediction],
            vocab,
            determined_numeric_feature_dim,
            max_seq_len=MAX_SEQ_LEN
        )

        # Wyodrębnij tensory dla enkodera z przetworzonego przykładu i przenieś na urządzenie
        encoder_token_ids_for_sample = collated_sample_for_prediction['encoder_token_ids'].to(device)
        encoder_numeric_features_for_sample = collated_sample_for_prediction['encoder_numeric_features'].to(device)
        encoder_attention_mask_for_sample = collated_sample_for_prediction['encoder_attention_mask'].to(device)

        question = example['question']
        original_answer = example['answer']

        # --- WAŻNA POPRAWKA: PRZEKAZANIE global_padded_feat_row ---
        predicted_answer_list = predict_and_decode_answer(
            model,
            encoder_token_ids_for_sample,
            encoder_numeric_features_for_sample,
            encoder_attention_mask_for_sample,
            vocab,
            device,
            padded_feat_row=global_padded_feat_row # <--- PRZEKAZUJEMY TUTAJ!
        )
        predicted_answer = predicted_answer_list[0] # predict_and_decode_answer zwraca listę, więc bierzemy pierwszy (i jedyny) element
        
        print(f"Q: {question}")
        print(f"A (Original): {original_answer}")
        print(f"A (Predicted): {predicted_answer}")
        print("-" * 20)
        # --- Koniec podglądu predykcji ---


    print("\nTraining complete.")

    # Plotting
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_ce_losses, label='Train CE Loss')
    plt.title('Training CE Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_num_losses, label='Train Numeric Loss', color='orange')
    plt.title('Training Numeric Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(val_exact_accuracies, label='Validation Exact Match Accuracy', color='green')
    plt.plot(val_numerical_accuracies, label='Validation Numerical Accuracy', color='red', linestyle='--')
    plt.title('Validation Accuracies Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    device = torch.device("cpu") # Change to "cuda" if GPU is available and set up
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")


    # --- KONFIGURACJA DANYCH I WYŚWIETLANIA ---
    # Ustaw na 1.0 (100%) dla pełnego treningu
    # Ustaw na np. 0.1 (10%) dla szybkiego testu
    data_sample_percentage = 0.1 # <--- ZMIEN TUTAJ, ABY KONTROLOWAĆ ROZMIAR DATASETU
    
    # Procent datasetu walidacyjnego używany do PEŁNEJ EWALUACJI (z dekodowaniem).
    # Ustaw na małą wartość (np. 0.05 lub 0.01) aby przyspieszyć evaluate().
    eval_sample_percentage = 0.05 # <--- NOWA ZMIENNA DLA KONTROLI SZYBKOSCI EWALUACJI

    # Liczba przykładów do wyświetlenia po każdej epoce. Ustawiono na 1 dla debugowania.
    num_examples_to_display = 1
    
    # Load dataset
    print("Loading dataset...")
    try:
        ds = load_dataset("openai/gsm8k", "main")
        train_dataset_full = ds['train']
        val_dataset_full = ds['test']
        print(f"Loaded full dataset: {len(train_dataset_full)} training examples, {len(val_dataset_full)} validation examples.")

        # Próbkowanie datasetu, jeśli data_sample_percentage < 1.0
        if data_sample_percentage < 1.0:
            train_size = int(len(train_dataset_full) * data_sample_percentage)
            val_size = int(len(val_dataset_full) * data_sample_percentage)
            
            # Wylosuj indeksy do próbkowania
            train_indices = random.sample(range(len(train_dataset_full)), train_size)
            val_indices_full_sample = random.sample(range(len(val_dataset_full)), val_size)

            train_dataset = train_dataset_full.select(train_indices)
            val_dataset_for_full_eval = val_dataset_full.select(val_indices_full_sample) # Used for full eval.
            print(f"Using sampled dataset: {len(train_dataset)} training examples, {len(val_dataset_for_full_eval)} validation examples for full eval ({data_sample_percentage*100:.0f}% of full).")
        else:
            train_dataset = train_dataset_full
            val_dataset_for_full_eval = val_dataset_full # Use full for full eval.
            print("Using full dataset for training and full evaluation.")

        # Próbkowanie dla SZYBKIEJ EWALUACJI (tylko do podglądu lub jeśli evaluate jest bardzo wolne)
        if eval_sample_percentage < 1.0:
            eval_size = max(1, int(len(val_dataset_for_full_eval) * eval_sample_percentage))
            eval_indices = random.sample(range(len(val_dataset_for_full_eval)), eval_size)
            val_dataset_for_evaluation = val_dataset_for_full_eval.select(eval_indices)
            print(f"Using sampled validation dataset for actual evaluation: {len(val_dataset_for_evaluation)} examples ({eval_sample_percentage*100:.0f}% of sampled validation).")
        else:
            val_dataset_for_evaluation = val_dataset_for_full_eval
            print("Using full sampled validation dataset for actual evaluation.")


    except Exception as e:
        print(f"Failed to load 'openai/gsm8k' dataset: {e}")
        print("Creating dummy dataset for demonstration purposes as a fallback.")
        dummy_data = {
            'question': [
                "What is 5 plus 3?", "Subtract 10 from 20.",
                "How many apples are there if you have 7 and get 2 more?",
                "What is the result of 15 minus 8?",
                "The temperature was 25 degrees Celsius and dropped by 5 degrees. What is the new temperature?",
                "If you have fifty-five and add twenty-three, what do you get?",
                "One hundred minus forty-two is what?",
                "Three times four is?", "What is half of 10?",
                "If you have 100 and spend 25, what's left?",
                "What is 10 plus 20 minus 5?", "How much is 10 times 2?",
                "The number 7 plus the number 8 equals?",
                "What is 12 divided by 3?", "Two hundred plus fifty is what?",
                "What is 9 minus 4?", "Add 1 to 10.", "Double 6.",
                "Half of 14.", "How much is 11 plus 11?",
            ],
            'answer': [
                "The answer is 8.", "It's 10.", "You have 9 apples.",
                "The result is 7.", "The new temperature is 20 degrees Celsius.",
                "You get 78.", "Fifty-eight.", "Twelve.", "Five.",
                "75 is left.", "25.", "20.", "15.", "4.",
                "Two hundred fifty.", "5.", "11.", "12.", "7.", "22.",
            ]
        }
        train_dataset = Dataset.from_dict(dummy_data)
        val_dataset_for_full_eval = Dataset.from_dict(dummy_data) # Renamed for clarity
        val_dataset_for_evaluation = Dataset.from_dict(dummy_data) # Used for evaluate() call
        ds = {'train': train_dataset, 'test': val_dataset_for_full_eval}


    print("Building vocabulary...")
    vocab = build_vocab_from_dataset(ds)
    vocab_size = len(vocab)
    idx_to_token = {idx: token for token, idx in vocab.items()} # Potrzebne do dekodowania
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens in vocab: <|num|> = {vocab.get('<|num|>')}, <|bos|> = {vocab.get('<|bos|>')}, <|eos|> = {vocab.get('<|eos|>')}, <|pad|> = {vocab.get('<|pad|>')}")

    sample_val = 123.0
    sample_type = 'int'
    determined_numeric_feature_dim = len(number_embedding_features(sample_val, sample_type))
    print(f"Determined numeric feature dimension: {determined_numeric_feature_dim}")

    MAX_SEQ_LEN = 128
    features_schema = get_features_schema(determined_numeric_feature_dim, MAX_SEQ_LEN)

    print("Initializing model...")
    model = ImprovedCrossEmbeddingSeq2SeqModel(
        vocab_size=vocab_size,
        token_dim=128,
        num_dim=128,
        hidden=256,
        encoder_layers=3,
        decoder_layers=3,
        dropout=0.1,
        feature_dim=determined_numeric_feature_dim
    ).to(device)

    print(f"Model initialized with vocab size: {vocab_size}, numeric feature input dim: {determined_numeric_feature_dim}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 20
    batch_size = 32
    gradient_accumulation_steps = 1

    num_training_steps = max(1, (len(train_dataset) // batch_size // gradient_accumulation_steps) * num_epochs)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_val_numerical_accuracy = -1.0
    train_ce_losses = []
    train_num_losses = []
    train_total_losses = []
    val_exact_accuracies = []
    val_numerical_accuracies = []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_ce_loss = 0.0
        total_numeric_loss = 0.0
        total_overall_loss = 0.0
        num_batches = 0

        # Wartość load_from_cache_file ustawiona na False w celu debugowania, zmień na True, gdy jesteś pewien, że preprocessing jest stabilny
        processed_train_dataset = train_dataset.map(
            lambda x: preprocess_example(x, vocab, determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN),
            remove_columns=['question', 'answer'],
            load_from_cache_file=True, # Zmienione na True
            desc="Preprocessing training examples",
            features=features_schema
        )

        train_dataloader = torch.utils.data.DataLoader(
            processed_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: custom_collate_fn(b, vocab, determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN)
        )

        optimizer.zero_grad()

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for i, batch in enumerate(train_dataloader):
                stats = train_step(model, optimizer, scheduler, batch, vocab, device, clip_grad=1.0)

                total_ce_loss += stats['ce_loss']
                total_numeric_loss += stats['numeric_loss']
                total_overall_loss += stats['total_loss']
                num_batches += 1

                pbar.set_postfix({
                    'CE': f"{stats['ce_loss']:.4f}",
                    'NumL': f"{stats['numeric_loss']:.4f}",
                    'TotalL': f"{stats['total_loss']:.4f}"
                })
                pbar.update(1)

        avg_ce_loss = total_ce_loss / num_batches
        avg_numeric_loss = total_numeric_loss / num_batches
        avg_total_loss = total_overall_loss / num_batches

        train_ce_losses.append(avg_ce_loss)
        train_num_losses.append(avg_numeric_loss)
        train_total_losses.append(avg_total_loss)

        print(f"\nEpoch {epoch+1} Avg Stats: CE: {avg_ce_loss:.4f}, Num Loss: {avg_numeric_loss:.4f}, Total Loss: {avg_total_loss:.4f}")

        # Evaluation
        print("Evaluating on validation set...")
        # Only preprocess the smaller sample for evaluation
        processed_val_dataset_for_evaluation = val_dataset_for_evaluation.map(
            lambda x: preprocess_example(x, vocab, determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN),
            remove_columns=['question', 'answer'],
            load_from_cache_file=True,
            desc="Preprocessing validation examples for evaluation",
            features=features_schema
        )

        val_exact_accuracy, val_numerical_accuracy = evaluate(
            model,
            processed_val_dataset_for_evaluation, # Użyj mniejszego datasetu tutaj
            vocab,
            device,
            batch_size=batch_size,
            collate_fn=lambda b: custom_collate_fn(b, vocab, determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN)
        )
        val_exact_accuracies.append(val_exact_accuracy)
        val_numerical_accuracies.append(val_numerical_accuracy)

        print(f"Validation Exact Match Accuracy: {val_exact_accuracy:.4f}")
        print(f"Validation Numerical Accuracy: {val_numerical_accuracy:.4f}")

        if val_numerical_accuracy > best_val_numerical_accuracy:
            best_val_numerical_accuracy = val_numerical_accuracy
            torch.save(model.state_dict(), "best_model_v3.pth")
            print(f"New best model saved with numerical accuracy: {best_val_numerical_accuracy:.4f}")

        # --- Podgląd predykcji po każdej epoce ---
        print("\n--- Model Predictions Sample ---")
        model.eval()
        # Wybieramy jeden losowy przykład z walidacyjnego datasetu do wyświetlenia
        sample_index = random.choice(range(len(val_dataset_for_full_eval))) # Wybierz jeden losowy indeks z PEŁNEGO próbkowanego walidacyjnego datasetu
        example = val_dataset_for_full_eval[sample_index] # Pobierz oryginalny przykład
        question = example['question']
        original_answer = example['answer']
        predicted_answer = predict_and_decode_answer(
            model, question, vocab, determined_numeric_feature_dim, MAX_SEQ_LEN, device
        )
        print(f"Q: {question}")
        print(f"A (Original): {original_answer}")
        print(f"A (Predicted): {predicted_answer}")
        print("-" * 20)
        # --- Koniec podglądu predykcji ---


    print("\nTraining complete.")

    # Plotting
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.plot(train_ce_losses, label='Train CE Loss')
    plt.title('Training CE Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_num_losses, label='Train Numeric Loss', color='orange')
    plt.title('Training Numeric Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(val_exact_accuracies, label='Validation Exact Match Accuracy', color='green')
    plt.plot(val_numerical_accuracies, label='Validation Numerical Accuracy', color='red', linestyle='--')
    plt.title('Validation Accuracies Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
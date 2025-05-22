import torch
import re
from tqdm import tqdm
from datasets import Dataset, Features, Value, Sequence

# Assuming blackhole.tokenizer and blackhole.embedding are correctly installed and accessible
from blackhole.tokenizer import tokenize as blackhole_tokenize # Rename to avoid conflict
from blackhole.embedding import number_embedding_features

# Import special tokens from config
from config import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN, SPECIAL_TOKENS
)

def preprocess_example(example, vocab, feature_embedding_dim, max_seq_len):
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
    pad_token_id = vocab.get(PAD_TOKEN, 0)
    unk_token_id = vocab.get(UNK_TOKEN, 0)
    num_token_id = vocab.get(NUM_TOKEN, unk_token_id)
    bos_token_id = vocab.get(BOS_TOKEN, unk_token_id)
    eos_token_id = vocab.get(EOS_TOKEN, unk_token_id)

    # padded_feat_row should be a list of floats, not a tensor, since we want lists of lists
    # Use a value that clearly indicates padding, e.g., -2.0, as Tanh output is [-1, 1]
    padded_feat_row_list = [-2.0] * feature_embedding_dim

    # --- Encoder Input Processing (Question) ---
    enc_tokens, enc_number_map = blackhole_tokenize(example['question'])
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
    ans_tokens, ans_number_map = blackhole_tokenize(example['answer'])

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
    enc_attention_mask = [False] * enc_len # False means not masked for PyTorch Transformer input_key_padding_mask
    if enc_len > max_seq_len:
        enc_token_ids = enc_token_ids[:max_seq_len]
        _enc_numeric_features = _enc_numeric_features[:max_seq_len]
        enc_attention_mask = enc_attention_mask[:max_seq_len]
    else:
        num_padding_needed_enc = max_seq_len - enc_len
        enc_token_ids += [pad_token_id] * num_padding_needed_enc
        _enc_numeric_features += [padded_feat_row_list] * num_padding_needed_enc
        enc_attention_mask += [True] * num_padding_needed_enc # True means masked


    # Decoder
    dec_len_after_slicing = len(decoder_input_token_ids)
    dec_attention_mask = [False] * dec_len_after_slicing # False means not masked

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
        dec_attention_mask += [True] * num_padding_needed_dec # True means masked

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

def custom_collate_fn(batch_examples, vocab, feature_embedding_dim, max_seq_len):
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
    all_tokens.update(SPECIAL_TOKENS) # Use special tokens from config

    # GSM8K has 'train' and 'test' splits
    for split_name in ['train', 'test']:
        if split_name in dataset:
            for example in tqdm(dataset[split_name], desc=f"Building vocab from {split_name} split"):
                q_tokens, _ = blackhole_tokenize(example['question']) # Use the correct tokenizer
                all_tokens.update(q_tokens)

                a_tokens, _ = blackhole_tokenize(example['answer']) # Use the correct tokenizer
                all_tokens.update(a_tokens)

    vocab = {tok: i for i, tok in enumerate(sorted(list(all_tokens)))}
    return vocab

def get_features_schema(feature_embedding_dim, max_seq_len):
    """
    Function to define the data schema for PyArrow.
    Features schema should reflect what preprocess_example RETURNS,
    which are Python lists of lists for numeric features.
    """
    return Features({
        "encoder_token_ids": Sequence(Value("int64"), length=max_seq_len),
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
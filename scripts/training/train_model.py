# File: train_model.py

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

# Hugging Face datasets and transformers
from datasets import load_dataset, Dataset, Features, Value, Sequence, concatenate_datasets
from transformers import get_scheduler
from torch.utils.data import DataLoader

# Set seeds for reproducibility from config (local import)
from config import RANDOM_SEED
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Import modularized components (local imports from scripts/training)
# Te importy są poprawne, ponieważ pliki są w tym samym katalogu co train_model.py
from config import (
    TOKEN_DIM, NUM_DIM, HIDDEN_DIM, ENCODER_LAYERS, DECODER_LAYERS, DROPOUT,
    MAX_SEQ_LEN, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    DATA_SAMPLE_PERCENTAGE, EVAL_SAMPLE_PERCENTAGE, NUM_EXAMPLES_TO_DISPLAY,
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN,
    CAP_TOKEN, ALLCAPS_TOKEN, SPACE_TOKEN, SPECIAL_TOKENS,
    DETERMINED_NUMERIC_FEATURE_DIM # <-- Ensure this is imported
)
from model import ImprovedCrossEmbeddingSeq2SeqModel
# REMOVED 'prepare_squad_example' from import as it's not defined in data_processing.py and not used
from data_processing import custom_collate_fn, preprocess_example, build_vocabulary, get_features_schema
from blackhole.nova.training import train_step
from blackhole.nova.evaluation import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and sample SQuAD dataset
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split='train')
    eval_dataset_full = load_dataset("squad", split='validation')

    if DATA_SAMPLE_PERCENTAGE < 1.0:
        sample_size_train = int(len(dataset) * DATA_SAMPLE_PERCENTAGE)
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(sample_size_train))
        print(f"Sampling {DATA_SAMPLE_PERCENTAGE:.2%} of training data ({len(dataset)} examples).")

    if EVAL_SAMPLE_PERCENTAGE < 1.0:
        sample_size_val = int(len(eval_dataset_full) * EVAL_SAMPLE_PERCENTAGE)
        val_dataset_for_full_eval = eval_dataset_full.shuffle(seed=RANDOM_SEED).select(range(sample_size_val))
        print(f"Sampling {EVAL_SAMPLE_PERCENTAGE:.2%} of validation data for full evaluation ({len(val_dataset_for_full_eval)} examples).")
    else:
        val_dataset_for_full_eval = eval_dataset_full

    print(f"Training dataset size after sampling: {len(dataset)}")
    print(f"Validation dataset size after sampling: {len(val_dataset_for_full_eval)}")

    # 2. Build Vocabulary
    print("Building vocabulary...")
    vocab = build_vocabulary(dataset, val_dataset_for_full_eval)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # NOWA LINIA: Utwórz odwrotną mapę indeksów do tokenów
    idx_to_token = {idx: token for token, idx in vocab.items()}

    # --- CRITICAL FIX START ---
    # The line below was causing the error by overriding the correct value from config.py
    # determined_numeric_feature_dim = NUM_DIM
    # We now directly use DETERMINED_NUMERIC_FEATURE_DIM imported from config.py

    # Use the value directly imported from config.py
    # No need to re-assign it here.
    print(f"Determined numeric feature dimension: {DETERMINED_NUMERIC_FEATURE_DIM}")
    # --- CRITICAL FIX END ---

    # 3. Initialize Model, Optimizer, Loss Functions
    model = ImprovedCrossEmbeddingSeq2SeqModel(
        vocab_size=vocab_size,
        token_dim=TOKEN_DIM,
        num_dim=NUM_DIM,
        hidden=HIDDEN_DIM,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        dropout=DROPOUT,
        feature_dim=DETERMINED_NUMERIC_FEATURE_DIM # Use the correct value from config.py
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    from blackhole.nova.loss_functions import focal_loss, mse_loss_for_numerical_features
    focal_loss_fn = focal_loss
    mse_loss_fn = mse_loss_for_numerical_features

    # 4. Preprocessing datasets
    print("Preprocessing datasets...")
    features_schema = get_features_schema(DETERMINED_NUMERIC_FEATURE_DIM) # Use the correct value

    # Konfiguracja train_dataset_processed
    train_dataset_processed = dataset.map(
        lambda example: (lambda res: {
            'encoder_token_ids': res.get('encoder_token_ids', []),
            'encoder_numeric_features': res.get('encoder_numeric_features', []),
            'encoder_attention_mask': res.get('encoder_attention_mask', []),
            'decoder_input_token_ids': res.get('decoder_input_token_ids', []),
            'decoder_input_numeric_features': res.get('decoder_input_numeric_features', []),
            'decoder_output_token_targets': res.get('decoder_output_token_targets', []),
            'decoder_output_numeric_targets': res.get('decoder_output_numeric_targets', []),
            'decoder_attention_mask': res.get('decoder_attention_mask', []),
            'original_question': res.get('original_question', ""),
            'original_answer': res.get('original_answer', ""),
            'original_numeric_values': res.get('original_numeric_values', []),
            'answer_token_type_map': res.get('answer_token_type_map', [])
        })(preprocess_example(
            {
                'question': [str(example['question'])],
                'answer': [str(example['answers']['text'][0])] if example['answers'] and 'text' in example['answers'] and len(example['answers']['text']) > 0 else ['']
            },
            vocab,
            DETERMINED_NUMERIC_FEATURE_DIM, # Use the correct value
            MAX_SEQ_LEN,
            idx_to_token
        )),
        batched=False,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )

    # Konfiguracja val_dataset_processed
    val_dataset_processed = val_dataset_for_full_eval.map(
        lambda example: (lambda res: {
            'encoder_token_ids': res.get('encoder_token_ids', []),
            'encoder_numeric_features': res.get('encoder_numeric_features', []),
            'encoder_attention_mask': res.get('encoder_attention_mask', []),
            'decoder_input_token_ids': res.get('decoder_input_token_ids', []),
            'decoder_input_numeric_features': res.get('decoder_input_numeric_features', []),
            'decoder_output_token_targets': res.get('decoder_output_token_targets', []),
            'decoder_output_numeric_targets': res.get('decoder_output_numeric_targets', []),
            'decoder_attention_mask': res.get('decoder_attention_mask', []),
            'original_question': res.get('original_question', ""),
            'original_answer': res.get('original_answer', ""),
            'original_numeric_values': res.get('original_numeric_values', []),
            'answer_token_type_map': res.get('answer_token_type_map', [])
        })(preprocess_example(
            {
                'question': [str(example['question'])],
                'answer': [str(example['answers']['text'][0])]
                             if example['answers'] and 'text' in example['answers'] and len(example['answers']['text']) > 0
                             else ['']
            },
            vocab,
            DETERMINED_NUMERIC_FEATURE_DIM, # Use the correct value
            MAX_SEQ_LEN,
            idx_to_token
        )),
        batched=False,
        num_proc=os.cpu_count(),
        remove_columns=val_dataset_for_full_eval.column_names,
        features=features_schema,
        load_from_cache_file=False
    )

    # FIXED: Set format for PyTorch, but keep ALL columns needed by collate_fn
    train_dataset_processed.set_format(type='torch', columns=[
        'encoder_token_ids', 'encoder_numeric_features', 'encoder_attention_mask',
        'decoder_input_token_ids', 'decoder_input_numeric_features',
        'decoder_output_token_targets', 'decoder_output_numeric_targets',
        'decoder_attention_mask', 'original_numeric_values', 'answer_token_type_map'
    ])
    val_dataset_processed.set_format(type='torch', columns=[
        'encoder_token_ids', 'encoder_numeric_features', 'encoder_attention_mask',
        'decoder_input_token_ids', 'decoder_input_numeric_features',
        'decoder_output_token_targets', 'decoder_output_numeric_targets',
        'decoder_attention_mask', 'original_numeric_values', 'answer_token_type_map'
    ])

    # 5. Tworzenie DataLoaders
    padded_feat_row = [-2.0] * DETERMINED_NUMERIC_FEATURE_DIM # Use the correct value
    train_dataloader = DataLoader(
        train_dataset_processed,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, vocab[PAD_TOKEN], padded_feat_row, MAX_SEQ_LEN, MAX_SEQ_LEN)
    )
    val_dataloader = DataLoader(
        val_dataset_processed,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, vocab[PAD_TOKEN], padded_feat_row, MAX_SEQ_LEN, MAX_SEQ_LEN)
    )

    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # 6. Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training")
        for batch_idx, batch in enumerate(progress_bar):
            loss = train_step(model, batch, optimizer, lr_scheduler, focal_loss_fn, mse_loss_fn, vocab, device)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_train_loss:.4f}")

        # 7. Evaluation
        print(f"Starting evaluation after Epoch {epoch+1}...")
        exact_match, numerical_acc = evaluate(
            model,
            val_dataset_processed,
            vocab,
            device,
            BATCH_SIZE,
            MAX_SEQ_LEN,
            collate_fn=lambda batch: custom_collate_fn(batch, vocab[PAD_TOKEN], [-2.0] * DETERMINED_NUMERIC_FEATURE_DIM, MAX_SEQ_LEN, MAX_SEQ_LEN)
        )
        print(f"Validation Exact Match: {exact_match:.2f}")
        print(f"Validation Numerical Accuracy: {numerical_acc:.2f}")

    # 8. Display example predictions
    print("\n--- Example Predictions (Validation Set) ---")
    num_samples_to_show = min(NUM_EXAMPLES_TO_DISPLAY, len(val_dataset_for_full_eval))
    sample_indices = random.sample(range(len(val_dataset_for_full_eval)), num_samples_to_show)

    from blackhole.nova.prediction import predict_and_decode_answer

    for i, idx in enumerate(sample_indices):
        original_question = val_dataset_for_full_eval[idx]['question']
        original_answer = val_dataset_for_full_eval[idx]['answers']['text'][0]

        single_example_processed = preprocess_example(
            {'question': [original_question], 'answer': [original_answer]},
            vocab,
            DETERMINED_NUMERIC_FEATURE_DIM, # <--- FIX THIS LINE
            MAX_SEQ_LEN,
            idx_to_token
        )

        # Ensure tensors are created correctly for single examples (batch_size=1)
        input_ids = torch.tensor([single_example_processed['encoder_token_ids']], dtype=torch.long).to(device)
        numeric_features = torch.tensor([single_example_processed['encoder_numeric_features']], dtype=torch.float32).to(device)
        attention_mask = torch.tensor([single_example_processed['encoder_attention_mask']], dtype=torch.bool).to(device)

        # Now predict_and_decode_answer returns two lists
        predicted_texts_batch, predicted_numbers_batch = predict_and_decode_answer(
            model,
            input_ids,
            numeric_features,
            attention_mask,
            vocab,
            device,
            MAX_SEQ_LEN,
            padded_feat_row=torch.full((DETERMINED_NUMERIC_FEATURE_DIM,), -2.0, dtype=torch.float32, device=device) # <--- AND FIX THIS LINE
        )

        # Get the single predicted text and number from the batch (since input_ids had batch_size=1)
        predicted_text = predicted_texts_batch[0]
        predicted_number = predicted_numbers_batch[0]

        print(f"\nPrzykład {i+1}:")
        print(f"Pytanie: {original_question}")
        print(f"Oryginalna odpowiedź: {original_answer}")
        print(f"Przewidziana odpowiedź: {predicted_text}")
        print(f"Przewidziana liczba (jeśli dotyczy): {predicted_number}")

    print("\nTraining complete.")
    model_save_path = "blackhole_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# This tokenize function from train_model.py is likely only used for initial vocabulary building
# It might be superseded by the one in data_processing.py
def tokenize(text):
    tokens = []
    number_map = {}
    idx = 0
    num_placeholder_count = 0

    number_pattern = r"[-+]?\d*\.\d+|\d+"

    last_end = 0
    for m in re.finditer(number_pattern, text):
        pre_num_text = text[last_end:m.start()]
        if pre_num_text:
            pre_num_tokens = re.findall(r'\b\w+\b|[^s\w]', pre_num_text)
            for t in pre_num_tokens:
                if t.strip(): tokens.append(t)

        num_val = float(m.group(0))
        tokens.append(NUM_TOKEN)
        number_map[len(tokens) - 1] = (num_val, 'float' if '.' in m.group(0) else 'int', m.group(0))
        last_end = m.end()

    remaining_text = text[last_end:]
    if remaining_text:
        remaining_tokens = re.findall(r'\b\w+\b|[^s\w]', remaining_text)
        for t in remaining_tokens:
            if t.strip(): tokens.append(t)

    final_tokens = []
    for token in tokens:
        if token.isupper() and len(token) > 1 and token not in SPECIAL_TOKENS:
            final_tokens.append(ALLCAPS_TOKEN)
            final_tokens.append(token.lower())
        elif token.istitle() and token not in SPECIAL_TOKENS:
            final_tokens.append(CAP_TOKEN)
            final_tokens.append(token.lower())
        elif token == ' ':
            final_tokens.append(SPACE_TOKEN)
        else:
            final_tokens.append(token)

    return final_tokens, number_map

from blackhole.nova.prediction import predict_and_decode_answer

if __name__ == "__main__":
    main()
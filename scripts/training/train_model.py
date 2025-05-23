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
from torch.utils.data import DataLoader

# Set base_dir for blackhole imports
try:
    # Corrected path: '..', '..' to go from 'scripts/training' to 'Blackhole-LLM'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..', '..'))
except NameError:
    base_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.insert(0, base_dir)

# Set seeds for reproducibility from config (local import)
from config import RANDOM_SEED
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Import modularized components (local imports from scripts/training)
from config import (
    TOKEN_DIM, NUM_DIM, HIDDEN_DIM, ENCODER_LAYERS, DECODER_LAYERS, DROPOUT,
    MAX_SEQ_LEN, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    DATA_SAMPLE_PERCENTAGE, EVAL_SAMPLE_PERCENTAGE, NUM_EXAMPLES_TO_DISPLAY,
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN, SPECIAL_TOKENS
)
from model import *
from data_processing import *
from inference import * # Assuming predict_and_decode_answer is here or handled internally by evaluate

# Import from your blackhole modules (explicit imports)
from blackhole.embedding import *
import blackhole.embedding
print(f"DEBUG: Załadowano embedding.py z: {blackhole.embedding.__file__}")

# Explicitly import functions from blackhole.nova submodules
from blackhole.nova import *


def load_and_prepare_datasets(data_sample_perc, eval_sample_perc, max_seq_len):
    """
    Loads the GSM8K dataset, applies sampling, and builds the vocabulary.
    Returns the sampled datasets and the vocabulary.
    """
    print("Loading dataset...")
    try:
        ds = load_dataset("openai/gsm8k", "main")
        train_dataset_full = ds['train']
        val_dataset_full = ds['test']
        print(f"Loaded full dataset: {len(train_dataset_full)} training examples, {len(val_dataset_full)} validation examples.")

        if data_sample_perc < 1.0:
            train_size = int(len(train_dataset_full) * data_sample_perc)
            val_size_for_full_sample = int(len(val_dataset_full) * data_sample_perc)
            
            train_indices = random.sample(range(len(train_dataset_full)), train_size)
            val_indices_full_sample = random.sample(range(len(val_dataset_full)), val_size_for_full_sample)

            train_dataset = train_dataset_full.select(train_indices)
            val_dataset_for_full_eval = val_dataset_full.select(val_indices_full_sample)
            print(f"Using sampled dataset: {len(train_dataset)} training examples, {len(val_dataset_for_full_eval)} validation examples for full eval ({data_sample_perc*100:.0f}% of full).")
        else:
            train_dataset = train_dataset_full
            val_dataset_for_full_eval = val_dataset_full
            print("Using full dataset for training and full evaluation.")

        if eval_sample_perc < 1.0:
            eval_size = max(1, int(len(val_dataset_for_full_eval) * eval_sample_perc))
            eval_indices = random.sample(range(len(val_dataset_for_full_eval)), eval_size)
            val_dataset_for_evaluation = val_dataset_for_full_eval.select(eval_indices)
            print(f"Using sampled validation dataset for actual evaluation: {len(val_dataset_for_evaluation)} examples ({eval_sample_perc*100:.0f}% of sampled validation).")
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
        val_dataset_for_full_eval = Dataset.from_dict(dummy_data)
        val_dataset_for_evaluation = Dataset.from_dict(dummy_data)
        ds = {'train': train_dataset, 'test': val_dataset_for_full_eval} # For vocab building

    print("Building vocabulary...")
    vocab = build_vocab_from_dataset(ds)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens in vocab: {NUM_TOKEN} = {vocab.get(NUM_TOKEN)}, {BOS_TOKEN} = {vocab.get(BOS_TOKEN)}, {EOS_TOKEN} = {vocab.get(EOS_TOKEN)}, {PAD_TOKEN} = {vocab.get(PAD_TOKEN)}")
    
    return train_dataset, val_dataset_for_evaluation, vocab

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare datasets and vocabulary
    train_dataset, val_dataset_for_evaluation, vocab = load_and_prepare_datasets(
        DATA_SAMPLE_PERCENTAGE, EVAL_SAMPLE_PERCENTAGE, MAX_SEQ_LEN
    )

    vocab_size = len(vocab)
    idx_to_token = {idx: token for token, idx in vocab.items()}

    # Determine numeric feature dimension (should be consistent across runs)
    sample_val = 123.0
    sample_type = 'int'
    determined_numeric_feature_dim = len(number_embedding_features(sample_val, sample_type))
    print(f"Determined numeric feature dimension: {determined_numeric_feature_dim}")

    features_schema = get_features_schema(determined_numeric_feature_dim, MAX_SEQ_LEN)

    print("Initializing model...")
    model = ImprovedCrossEmbeddingSeq2SeqModel(
        vocab_size=vocab_size,
        token_dim=TOKEN_DIM,
        num_dim=NUM_DIM,
        hidden=HIDDEN_DIM,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        dropout=DROPOUT,
        feature_dim=determined_numeric_feature_dim
    ).to(device)

    print("Preprocessing datasets with Hugging Face map...")
    processed_train_dataset = train_dataset.map(
        lambda ex: preprocess_example(ex, vocab, determined_numeric_feature_dim, MAX_SEQ_LEN),
        batched=False,
        remove_columns=train_dataset.column_names,
        features=features_schema
    )
    processed_val_dataset = val_dataset_for_evaluation.map(
        lambda ex: preprocess_example(ex, vocab, determined_numeric_feature_dim, MAX_SEQ_LEN),
        batched=False,
        remove_columns=val_dataset_for_evaluation.column_names,
        features=features_schema
    )
    print(f"Processed training dataset size: {len(processed_train_dataset)}")
    print(f"Processed validation dataset size: {len(processed_val_dataset)}")


    train_dataloader = DataLoader(
        processed_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: custom_collate_fn(b, vocab, determined_numeric_feature_dim, MAX_SEQ_LEN)
    )
    eval_dataloader = DataLoader(
        processed_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: custom_collate_fn(b, vocab, determined_numeric_feature_dim, MAX_SEQ_LEN)
    )

    print("Initializing optimizer and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        model.train() # Set model to training mode
        total_train_loss = 0
        # Iterate over batches from train_dataloader
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            # Pass individual batch to train_step, along with vocab and device
            batch_loss_dict = train_step(model, batch, optimizer, lr_scheduler, focal_loss, mse_loss_for_numerical_features, vocab, device)
            total_train_loss += batch_loss_dict['total_loss']

        train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

        # Evaluation step
        model.eval() # Set model to evaluation mode
        val_exact_acc, val_numerical_acc = evaluate(
            model, processed_val_dataset, vocab, device,
            batch_size=BATCH_SIZE,
            max_decoding_len=MAX_SEQ_LEN,
            collate_fn=lambda b: custom_collate_fn(b, vocab, determined_numeric_feature_dim, MAX_SEQ_LEN)
        )
        print(f"Epoch {epoch+1} Validation Exact Match: {val_exact_acc:.4f}, Numerical Accuracy: {val_numerical_acc:.4f}")

        print("\n--- Sample Predictions (Run inference separately if needed) ---")

    print("Training complete!")
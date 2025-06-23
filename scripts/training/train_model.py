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
from datasets import load_dataset, Dataset, Features, Value, Sequence, concatenate_datasets
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
from model import * # Assuming 'model.py' contains your model definition (e.g., ImprovedCrossEmbeddingSeq2SeqModel)
from data_processing import * # Import all from data_processing
from inference import * # Import from your blackhole modules (explicit imports)
import blackhole.embedding
print(f"DEBUG: Załadowano embedding.py z: {blackhole.embedding.__file__}")

# Explicitly import functions from blackhole.nova submodules
# Assuming these exist in your project structure for loss functions.
# If not, you might need to define them or adjust imports.
try:
    from blackhole.nova import focal_loss, mse_loss_for_numerical_features, train_step
except ImportError:
    print("WARNING: Could not import focal_loss, mse_loss_for_numerical_features, train_step from blackhole.nova. Using dummy functions.")
    # Dummy implementations if imports fail, for demonstration purposes
    def focal_loss(output, target, gamma=2.0, alpha=0.25):
        # Dummy Focal Loss (replace with actual implementation)
        ce_loss = F.cross_entropy(output, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (alpha * (1 - pt)**gamma * ce_loss).mean()
        return focal_loss

    def mse_loss_for_numerical_features(output_numeric_features, target_numeric_features, attention_mask):
        # Dummy MSE Loss for Numerical Features (replace with actual implementation)
        # Only consider positions where attention_mask is True (i.e., not padding)
        masked_output = output_numeric_features * attention_mask.unsqueeze(-1)
        masked_target = target_numeric_features * attention_mask.unsqueeze(-1)
        return F.mse_loss(masked_output, masked_target, reduction='mean')

    def train_step(model, batch, optimizer, lr_scheduler, focal_loss_fn, mse_loss_fn, vocab, device):
        # Dummy train_step (replace with actual implementation)
        model.train()
        optimizer.zero_grad()

        input_ids = batch['encoder_token_ids'].to(device)
        numeric_features = batch['encoder_numeric_features'].to(device)
        encoder_attention_mask = batch['encoder_attention_mask'].to(device)
        decoder_input_token_ids = batch['decoder_input_token_ids'].to(device)
        decoder_input_numeric_features = batch['decoder_input_numeric_features'].to(device)
        decoder_output_token_targets = batch['decoder_output_token_targets'].to(device)
        decoder_output_numeric_targets = batch['decoder_output_numeric_targets'].to(device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(device) # Mask for decoder targets

        predicted_output_ids, predicted_numeric_features = model(
            encoder_token_ids=input_ids,
            encoder_numeric_features=numeric_features,
            decoder_token_ids=decoder_input_token_ids, # Decoder input for auto-regressive generation
            decoder_numeric_features_input=decoder_input_numeric_features,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask # Pass decoder mask to model
        )

        # Reshape for loss calculation
        # For token prediction (classification), flatten output and target
        token_loss = focal_loss_fn(
            predicted_output_ids.view(-1, model.vocab_size), # Reshape to (batch_size * seq_len, vocab_size)
            decoder_output_token_targets.view(-1) # Reshape to (batch_size * seq_len)
        )

        # For numerical prediction (regression), apply MSE
        numerical_loss = mse_loss_fn(
            predicted_numeric_features,
            decoder_output_numeric_targets,
            decoder_attention_mask # Apply mask for numerical loss
        )

        total_loss = token_loss + numerical_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients to prevent exploding gradients
        optimizer.step()
        lr_scheduler.step()

        return {'total_loss': total_loss.item(), 'token_loss': token_loss.item(), 'numerical_loss': numerical_loss.item()}



def load_and_prepare_datasets(data_sample_perc, eval_sample_perc, max_seq_len):
    """
    Loads desired datasets, applies sampling, and builds the vocabulary.
    Returns the sampled datasets and the vocabulary.
    """
    print("Loading datasets...")
    all_raw_datasets = {}

    # --- 1. Load GSM8K dataset ---
    try:
        ds_gsm8k = load_dataset("openai/gsm8k", "main")
        all_raw_datasets['gsm8k_train'] = ds_gsm8k['train']
        all_raw_datasets['gsm8k_test'] = ds_gsm8k['test']
        print(f"Loaded GSM8K: {len(ds_gsm8k['train'])} training examples, {len(ds_gsm8k['test'])} test examples.")
    except Exception as e:
        print(f"WARNING: Failed to load 'openai/gsm8k' dataset: {e}")
        # Fallback to dummy data if GSM8K fails
        dummy_data = {
            'question': ["What is 5 plus 3?", "Subtract 10 from 20."],
            'answer': ["The answer is 8.", "It's 10."],
        }
        all_raw_datasets['gsm8k_train'] = Dataset.from_dict(dummy_data)
        all_raw_datasets['gsm8k_test'] = Dataset.from_dict(dummy_data)
        print("Using dummy data for GSM8K.")

    # --- 2. Load and process additional datasets (example: SQuAD) ---
    # SQuAD dataset structure: 'id', 'title', 'context', 'question', 'answers'
    # We need to transform it to 'question', 'answer'
    try:
        ds_squad = load_dataset("squad")
        
        # Function to flatten SQuAD answers and contexts into a single question-answer pair
        def process_squad_example(example):
            answer_text = example['answers']['text'][0] if example['answers']['text'] else ""
            question = example['question']
            
            return {'question': question, 'answer': answer_text}

        all_raw_datasets['squad_train'] = ds_squad['train'].map(process_squad_example, remove_columns=ds_squad['train'].column_names)
        all_raw_datasets['squad_validation'] = ds_squad['validation'].map(process_squad_example, remove_columns=ds_squad['validation'].column_names)
        print(f"Loaded SQuAD: {len(all_raw_datasets['squad_train'])} training examples, {len(all_raw_datasets['squad_validation'])} validation examples.")

    except Exception as e:
        print(f"WARNING: Failed to load 'squad' dataset: {e}. Skipping SQuAD.")
        # Add dummy data to avoid errors if SQuAD loading fails
        all_raw_datasets['squad_train'] = Dataset.from_dict({'question': ["What is the capital of France?"], 'answer': ["Paris."]})
        all_raw_datasets['squad_validation'] = Dataset.from_dict({'question': ["What is the capital of France?"], 'answer': ["Paris."]})


    # --- Combine all training and validation datasets ---
    # Filter out empty datasets from the dictionary before concatenation
    train_datasets_list = [d for k, d in all_raw_datasets.items() if 'train' in k and d is not None]
    test_datasets_list = [d for k, d in all_raw_datasets.items() if ('test' in k or 'validation' in k) and d is not None]

    if not train_datasets_list or not test_datasets_list:
        raise RuntimeError("No datasets were successfully loaded for training or validation.")

    train_dataset_full = concatenate_datasets(train_datasets_list)
    val_dataset_full = concatenate_datasets(test_datasets_list)

    print(f"Combined full dataset: {len(train_dataset_full)} training examples, {len(val_dataset_full)} validation examples.")

    # Apply sampling based on config (if percentages are less than 1.0)
    if data_sample_perc < 1.0:
        train_size = int(len(train_dataset_full) * data_sample_perc)
        val_size_for_full_sample = int(len(val_dataset_full) * data_sample_perc) # This might be large for sampling
        
        train_indices = random.sample(range(len(train_dataset_full)), train_size)
        val_indices_full_sample = random.sample(range(len(val_dataset_full)), val_size_for_full_sample)

        train_dataset = train_dataset_full.select(train_indices)
        val_dataset_for_full_eval = val_dataset_full.select(val_indices_full_sample)
        print(f"Using sampled dataset: {len(train_dataset)} training examples, {len(val_dataset_for_full_eval)} validation examples for full eval ({data_sample_perc*100:.0f}% of full).")
    else:
        train_dataset = train_dataset_full
        val_dataset_for_full_eval = val_dataset_full # Use full combined validation dataset
        print("Using full combined dataset for training and full evaluation.")

    # Smaller sample for actual evaluation during training loops to save time
    if eval_sample_perc < 1.0:
        eval_size = max(1, int(len(val_dataset_for_full_eval) * eval_sample_perc))
        eval_indices = random.sample(range(len(val_dataset_for_full_eval)), eval_size)
        val_dataset_for_evaluation = val_dataset_for_full_eval.select(eval_indices)
        print(f"Using sampled validation dataset for actual evaluation: {len(val_dataset_for_evaluation)} examples ({eval_sample_perc*100:.0f}% of sampled validation).")
    else:
        val_dataset_for_evaluation = val_dataset_for_full_eval # Use full combined validation dataset
        print("Using full sampled validation dataset for actual evaluation.")

    # Build vocabulary from the combined full training and validation datasets
    # Pass all relevant datasets for vocab building to ensure comprehensive vocab
    # It's usually best to build vocab from training data to avoid data leakage.
    print("Building vocabulary from combined training dataset...")
    vocab = build_vocab_from_dataset({'train': train_dataset_full}) # Build vocab from the FULL combined training set
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens in vocab: {NUM_TOKEN} = {vocab.get(NUM_TOKEN)}, {BOS_TOKEN} = {vocab.get(BOS_TOKEN)}, {EOS_TOKEN} = {vocab.get(EOS_TOKEN)}, {PAD_TOKEN} = {vocab.get(PAD_TOKEN)}")
    
    return train_dataset, val_dataset_for_evaluation, vocab


def validate_epoch(model, dataloader, vocab, device, max_decoding_len, numeric_feature_dim):
    model.eval()
    total_exact_match = 0
    total_numerical_match = 0
    total_samples = 0
    
    idx_to_token = {idx: token for token, idx in vocab.items()}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            encoder_token_ids = batch['encoder_token_ids'].to(device)
            encoder_numeric_features = batch['encoder_numeric_features'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            
            original_answers_text = batch['original_answers_text']
            original_numeric_values = batch['original_numeric_values']

            # Generate predictions for the entire batch at once
            predicted_texts, predicted_numbers, _ = predict_and_decode_answer(
                model, 
                encoder_token_ids,
                encoder_numeric_features,
                encoder_attention_mask,
                vocab, 
                device, 
                max_decoding_len, 
                numeric_feature_dim 
            )

            # Iterate over single examples in the batch for evaluation
            for i in range(len(predicted_texts)): 
                # Get ground truth for the current example
                target_text = original_answers_text[i]
                target_numbers_for_example = original_numeric_values[i]

                # Exact Match
                cleaned_predicted_text = re.sub(r'\s+', ' ', predicted_texts[i]).strip()
                cleaned_target_text = re.sub(r'\s+', ' ', target_text).strip()

                if cleaned_predicted_text == cleaned_target_text:
                    total_exact_match += 1

                # Numerical Accuracy
                current_predicted_number = predicted_numbers[i]
                if target_numbers_for_example and current_predicted_number is not None:
                    target_value = target_numbers_for_example[-1] 
                    if math.isclose(current_predicted_number, target_value, rel_tol=1e-5, abs_tol=1e-5):
                        total_numerical_match += 1
                elif not target_numbers_for_example and current_predicted_number is None:
                    total_numerical_match += 1
                
            total_samples += encoder_token_ids.size(0)

    exact_acc = total_exact_match / total_samples if total_samples > 0 else 0
    numerical_acc = total_numerical_match / total_samples if total_samples > 0 else 0
    
    return exact_acc, numerical_acc


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare datasets and vocabulary
    train_dataset, val_dataset_for_full_eval, vocab = load_and_prepare_datasets(
        DATA_SAMPLE_PERCENTAGE, EVAL_SAMPLE_PERCENTAGE, MAX_SEQ_LEN
    )

    # Determine numeric feature dimension FIRST
    sample_val = 123.0
    sample_type = 'int'
    determined_numeric_feature_dim = len(blackhole.embedding.number_embedding_features(sample_val, sample_type))
    print(f"Determined numeric feature dimension: {determined_numeric_feature_dim}")

    # Now define features_schema (after numeric_feature_dim is known)
    features_schema = get_features_schema(determined_numeric_feature_dim)
    
    vocab_size = len(vocab)
    idx_to_token = {idx: token for token, idx in vocab.items()}

    # Initialize model after determining vocab_size and numeric_feature_dim
    # Corrected model class name from EncoderDecoder to ImprovedCrossEmbeddingSeq2SeqModel
    model = ImprovedCrossEmbeddingSeq2SeqModel(
        vocab_size=vocab_size,
        token_dim=TOKEN_DIM,
        num_dim=NUM_DIM,
        hidden=HIDDEN_DIM,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        dropout=DROPOUT,
        feature_dim=determined_numeric_feature_dim # Pass the determined feature_dim to the model
    ).to(device)
    print("Model initialized.")

    print("Preprocessing datasets with Hugging Face map...")


    # Create a partial function for custom_collate_fn to pass to DataLoader
    # This avoids the pickling error on Windows by not using a lambda
    from functools import partial
    
    # Set num_workers to 0 on Windows to avoid PicklingError with lambda in collate_fn
    # For other OS (Linux/macOS), num_workers can be > 0.
    num_dataloader_workers = 0 
    if os.name == 'posix': # Check if OS is not Windows (i.e., Linux or macOS)
        num_dataloader_workers = os.cpu_count() // 2 if os.cpu_count() else 0

    train_dataloader = DataLoader(
        processed_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(custom_collate_fn, vocab=vocab, numeric_feature_dim=determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN),
        num_workers=num_dataloader_workers
    )
    eval_dataloader = DataLoader(
        processed_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(custom_collate_fn, vocab=vocab, numeric_feature_dim=determined_numeric_feature_dim, max_seq_len=MAX_SEQ_LEN),
        num_workers=num_dataloader_workers
    )

    print("Initializing optimizer and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    best_val_accuracy = -1.0
    best_model_path = "best_model.pth"

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            batch_loss_dict = train_step(model, batch, optimizer, lr_scheduler, focal_loss, mse_loss_for_numerical_features, vocab, device)
            total_train_loss += batch_loss_dict['total_loss']

        train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved to {best_model_path} (from end of Epoch {epoch+1})")


    print("Training complete!")

    # --- Final Evaluation with Best Model ---
    print(f"\n--- Final Evaluation with Best Model ({best_model_path}) ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print(f"WARNING: Best model not found at {best_model_path}. Using the last epoch's model.")
    model.eval()

    final_exact_acc, final_numerical_acc = validate_epoch(
        model, eval_dataloader, vocab, device,
        max_decoding_len=MAX_SEQ_LEN,
        numeric_feature_dim=determined_numeric_feature_dim
    )
    print(f"Final Validation Exact Match: {final_exact_acc:.4f}, Numerical Accuracy: {final_numerical_acc:.4f}")

    # --- Generowanie przykładowych odpowiedzi ---
    print("\n--- Generowanie przykładowych odpowiedzi ---")
    
    num_samples_to_show = NUM_EXAMPLES_TO_DISPLAY
    sample_indices = random.sample(range(len(val_dataset_for_full_eval)), min(num_samples_to_show, len(val_dataset_for_full_eval)))

    for i, idx in enumerate(sample_indices):
        original_question = val_dataset_for_full_eval[idx]['question']
        original_answer = val_dataset_for_full_eval[idx]['answer']

        single_example_processed = preprocess_example(
            {'question': [original_question], 'answer': [original_answer]}, 
            vocab,
            determined_numeric_feature_dim,
            MAX_SEQ_LEN
        )

        # Extract the single example from the batched output
        input_ids = torch.tensor([single_example_processed['encoder_token_ids'][0]], dtype=torch.long).to(device)
        numeric_features = torch.tensor([single_example_processed['encoder_numeric_features'][0]], dtype=torch.float32).to(device)
        attention_mask = torch.tensor([single_example_processed['encoder_attention_mask'][0]], dtype=torch.bool).to(device)

        predicted_text, predicted_number, _ = predict_and_decode_answer(
            model,
            input_ids,
            numeric_features,
            attention_mask,
            vocab,
            device,
            MAX_SEQ_LEN,
            determined_numeric_feature_dim
        )

        print(f"\nPrzykład {i+1}:")
        print(f"Pytanie: {original_question}")
        print(f"Prawidłowa odpowiedź: {original_answer}")
        print(f"Wygenerowana odpowiedź (tekst): {predicted_text}")
        print(f"Wygenerowana odpowiedź (liczba): {predicted_number}")
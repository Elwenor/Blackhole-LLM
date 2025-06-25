import os, sys
os.environ['USE_TF'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import transformers
transformers.utils.logging.set_verbosity_error()

# Zmieniamy sposób ładowania datasetu
from datasets import load_dataset, Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
import torch
from tokenizers.processors import TemplateProcessing # Potrzebne do train_tokenizer

# Ustawienie PROJECT_ROOT w sposób spójny z run_pretraining_mlm.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import naszych niestandardowych komponentów
import scripts.training.config as config
from scripts.training.data_processing import Seq2SeqTextNumericDataset
from scripts.training.model import BlackholeSeq2SeqForConditionalGeneration

# Import komponentów z TWOJEGO nowego pakietu nova_test_file
from nova_test_file.configuration_nova import BlackholeConfig
from nova_test_file.hugging_tokenizer2 import BlackholeTokenizer2 # Changed to import class directly
from nova_test_file.hugging_tokenizer2 import CUSTOM_SPECIAL_TOKENS
from nova_test_file.data_collator2 import BlackholeDataCollatorForSeq2Seq
from nova_test_file.nova_trainer2 import NovaTrainer # CORRECTED: Using the actual class name 'NovaTrainer' from nova_trainer2.py
from nova_test_file.hugging_tokenizer2 import Tokenizer, models, pre_tokenizers, trainers, decoders # Import necessary classes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_tokenizer(corpus_iterator, vocab_size, save_path, special_tokens):
    """
    Trains a BlackholeTokenizer2 from a corpus and saves it.
    """
    logger.info(f"Training new tokenizer with vocab size {vocab_size}...")
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Split(Regex(r"[0-9]+(?:[\.,][0-9]+)?"), "isolated"),
        pre_tokenizers.Split(Regex(r"[^\w\s]"), "isolated"),
    ])

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    # Add post-processor and decoder
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.decoder = decoders.WordPiece()

    # Save the trained tokenizer
    os.makedirs(save_path, exist_ok=True)
    tokenizer_file_path = os.path.join(save_path, "tokenizer.json")
    tokenizer.save(tokenizer_file_path)
    logger.info(f"Trained tokenizer saved to {tokenizer_file_path}")

    # Also save the vocab.json for compatibility with from_pretrained
    vocab_file_path = os.path.join(save_path, "vocab.json")
    with open(vocab_file_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)
    logger.info(f"Vocab saved to {vocab_file_path}")

    return tokenizer_file_path


def main():
    logger.info("--- Starting Blackhole LLM training script ---")
    set_seed(config.RANDOM_SEED)

    # 1. Device configuration
    logger.info(f"Using device: {config.DEVICE}")

    # 2. Load Dataset from Hugging Face Hub
    # Będziemy używać datasetu 'gsm8k'
    logger.info("Loading dataset 'gsm8k' from Hugging Face Hub...")
    try:
        # Load the dataset. gsm8k has 'train' and 'test' splits. We'll use 'test' as validation.
        raw_datasets = load_dataset("gsm8k", "main") # Loads with 'train' and 'test' splits
        
        # Preprocess the dataset: rename 'question' to 'text' and 'answer' to 'labels'
        # This makes it compatible with Seq2SeqTextNumericDataset and typical Seq2Seq setups.
        def preprocess_function(examples):
            # Combine question and answer for tokenizer training corpus
            # For Seq2Seq, 'text' will be the input, 'labels' the target
            examples["text"] = [q for q in examples["question"]]
            examples["labels"] = [a for a in examples["answer"]]
            return examples

        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=["question", "answer"] # POPRAWIONO: Usunięto 'idx'
        )

        train_dataset_hf = raw_datasets["train"]
        eval_dataset_hf = raw_datasets["test"] # Using 'test' split as validation

        logger.info(f"Dataset 'gsm8k' loaded. Train split size: {len(train_dataset_hf)}, Eval split size: {len(eval_dataset_hf)}")

    except Exception as e:
        logger.error(f"Error loading dataset from Hugging Face: {e}")
        sys.exit(1)

    # 3. Load or Train Tokenizer
    tokenizer_file_path = os.path.join(config.TOKENIZER_PATH, "tokenizer.json")
    if not os.path.exists(tokenizer_file_path):
        logger.info("Tokenizer not found. Training a new one.")
        # Create an iterator over the 'text' column for tokenizer training
        corpus_iterator = (example["text"] for example in train_dataset_hf)
        special_tokens = list(CUSTOM_SPECIAL_TOKENS.values())
        
        trained_tokenizer_file = train_tokenizer(
            corpus_iterator=corpus_iterator,
            vocab_size=config.VOCAB_SIZE,
            save_path=config.TOKENIZER_PATH,
            special_tokens=special_tokens
        )
        # Load the newly trained tokenizer
        tokenizer = BlackholeTokenizer2.from_pretrained(config.TOKENIZER_PATH, numeric_feature_size=config.DETERMINED_NUMERIC_FEATURE_DIM)
        logger.info(f"New tokenizer loaded from {config.TOKENIZER_PATH}")
    else:
        logger.info("Loading existing tokenizer...")
        tokenizer = BlackholeTokenizer2.from_pretrained(config.TOKENIZER_PATH, numeric_feature_size=config.DETERMINED_NUMERIC_FEATURE_DIM)
        logger.info("Tokenizer loaded.")

    # Upewnij się, że tokenizator ma ustawiony pad_token_id, jeśli model tego wymaga
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.token_to_id('[PAD]')

    # 4. Prepare Datasets for Model Training
    # Seq2SeqTextNumericDataset będzie potrzebował 'text' i 'labels'
    train_dataset = Seq2SeqTextNumericDataset(
        dataset=train_dataset_hf,
        tokenizer=tokenizer,
        data_percentage=config.DATA_SAMPLE_PERCENTAGE,
        max_seq_len=config.MAX_SEQ_LEN
    )
    eval_dataset = Seq2SeqTextNumericDataset(
        dataset=eval_dataset_hf,
        tokenizer=tokenizer,
        data_percentage=config.EVAL_SAMPLE_PERCENTAGE,
        max_seq_len=config.MAX_SEQ_LEN
    )

    logger.info(f"Train dataset prepared, size: {len(train_dataset)}")
    logger.info(f"Eval dataset prepared, size: {len(eval_dataset)}")

    # 5. Initialize Model
    logger.info("Initializing Blackhole model configuration...")
    model_config = BlackholeConfig(
        vocab_size=tokenizer.vocab_size, # Użyj rzeczywistego rozmiaru słownika tokenizera
        hidden_dim=config.HIDDEN_DIM,
        encoder_layers=config.ENCODER_LAYERS,
        decoder_layers=config.DECODER_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        intermediate_size=config.INTERMEDIATE_SIZE,
        dropout=config.DROPOUT,
        type_vocab_size=config.TYPE_VOCAB_SIZE,
        layer_norm_eps=config.LAYER_NORM_EPS,
        numeric_feature_dims=config.NUMERIC_FEATURE_DIMS,
        numeric_embedding_fusion_type=config.NUMERIC_EMBEDDING_FUSION_TYPE,
        numeric_projection_intermediate_size_ratio=config.NUMERIC_PROJECTION_INTERMEDIATE_SIZE_RATIO,
        numeric_heavy_feature_freeze=config.NUMERIC_HEAVY_FEATURE_FREEZE,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id, # Typically CLS for encoder-decoder start
        eos_token_id=tokenizer.sep_token_id, # Typically SEP for encoder-decoder end
    )
    logger.info(f"Model config: {model_config}")

    logger.info("Initializing BlackholeSeq2SeqForConditionalGeneration model...")
    model = BlackholeSeq2SeqForConditionalGeneration(model_config)
    model.to(config.DEVICE)
    logger.info("Model initialized.")

    # 6. Initialize Data Collator
    logger.info("Initializing Data Collator...")
    data_collator = BlackholeDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model, # Pass model for dynamic padding
        padding="max_length",
        max_length=config.MAX_SEQ_LEN,
        return_tensors="pt"
    )
    logger.info("Data Collator initialized.")

    # 7. Initialize Training Arguments and Trainer
    logger.info("Initializing Training Arguments and Trainer...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,
        fp16=config.DEVICE == "cuda", # Enable FP16 only if CUDA is available
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True, # Important for Seq2Seq models
        report_to="none", # Disable reporting to external services like W&B unless configured
    )

    trainer = NovaTrainer( 
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 8. Start Training
    logger.info("--- All components initialized. Starting training... ---")
    try:
        trainer.train()
        logger.info("Training complete.")

        # 9. Save the final model
        final_model_path = os.path.join(config.OUTPUT_DIR, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path) # Save tokenizer with the model
        logger.info(f"Final model and tokenizer saved to {final_model_path}")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
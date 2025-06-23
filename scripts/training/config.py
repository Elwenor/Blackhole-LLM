# File: config.py
import os

# Set seeds for reproducibility
RANDOM_SEED = 42

# Model Hyperparameters
TOKEN_DIM = 128
NUM_DIM = 128
HIDDEN_DIM = 256
ENCODER_LAYERS = 3
DECODER_LAYERS = 3
DROPOUT = 0.1
MAX_SEQ_LEN = 128

# Training Hyperparameters
NUM_EPOCHS = 1
# [POPRAWKA] Zmniejsz BATCH_SIZE, jeśli wystąpi błąd "Out of Memory" na GPU z małą ilością VRAM.
# Dla GTX 650 bezpieczniej zacząć od 16 lub 32.
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Data Loading and Sampling
DATA_SAMPLE_PERCENTAGE = 0.2
EVAL_SAMPLE_PERCENTAGE = 0.2

# Display and Debug
NUM_EXAMPLES_TO_DISPLAY = 3 # Number of examples to show predictions for after each epoch

# Special Tokens
PAD_TOKEN = '<|pad|>'
UNK_TOKEN = '<|unk|>'
BOS_TOKEN = '<|bos|>'
EOS_TOKEN = '<|eos|>'
CAP_TOKEN = '<|cap|>'
ALLCAPS_TOKEN = '<|allcaps|>'
NUM_TOKEN = '<|num|>'
SPACE_TOKEN = '<|space|>'

SPECIAL_TOKENS = {
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
    CAP_TOKEN, ALLCAPS_TOKEN, NUM_TOKEN, SPACE_TOKEN
}
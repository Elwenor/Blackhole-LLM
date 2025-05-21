import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blackhole.tokenizer import *
from blackhole.embedding import *

text = """
During the trial, the compound concentration peaked at 1.23456789e+10 mol/L, but dropped to nearly zero (0.000000000045) within 3.14 seconds. 
Patient survival improved by 123456789%, while the control group had an incidence rate of -0.0000001. 
The total cost exceeded $9,876,543.21, yet the efficiency was measured at 0.0000000000001 units per cycle. 
Unexpectedly, a sample showed a negative reading of -987654321, which contradicted the baseline of 0.0000000001.
"""

# Tokenize input text
tokens, number_map = tokenize(text)
print("Tokens:", tokens)
print("\nNumber Map:")
for idx, (val, typ, raw) in number_map.items():
    print(f"{idx}: {val} ({typ}) -> '{raw}'")

# Prepare inputs: token_ids is a tensor [1, L]
token_ids, num_map, vocab = prepare_inputs(tokens, number_map)

# Initialize embedding models
token_emb_model = TokenEmbedding(vocab_size=len(vocab))
number_emb_model = NumberEmbedding()

# Embed tokens
token_embeddings = token_emb_model(token_ids)  # shape: [1, L, token_emb_dim]

B, L = token_ids.shape

# Create tensor for numeric features [B, L, feature_dim], fill zeros by default
raw_feats = torch.zeros(B, L, 12)  # 12 is the assumed feature dimension from number_embedding_features

# Map numeric values to features in the batch (here batch=1)
for token_idx, (val, typ, raw) in num_map.items():
    # token_idx is the position in the token list / token_ids
    # Fill the numeric features for batch 0, position token_idx
    raw_feats[0, token_idx] = number_embedding_features(val, typ)

# Embed numeric features
number_embeddings = number_emb_model(raw_feats)  # shape: [1, L, number_emb_dim]

print("\n--- Numeric token embeddings (<num>) ---")
for i in range(L):
    token = tokens[i]
    if i in num_map:
        token_emb = token_embeddings[0, i]
        number_emb = number_embeddings[0, i]
        print(f"Token index {i}: '{token}'")
        print(f"Token embedding: {token_emb.detach().cpu().numpy()}")
        print(f"Number embedding: {number_emb.detach().cpu().numpy()}")
        print("-" * 50)

print("\n--- Non-numeric token embeddings (ignoring spaces and <|space|>) ---")
count = 0
max_print = 50
i = 0

while count < max_print and i < L:
    token = tokens[i]
    if i in num_map or token.strip() == "" or token == "<|space|>":
        i += 1
        continue  # skip numbers and spaces

    token_emb = token_embeddings[0, i]
    print(f"Token index {i}: '{token}'")
    print(f"Token embedding: {token_emb.detach().cpu().numpy()}")
    print("Number embedding: None (token is not numeric)")
    print("-" * 50)

    count += 1
    i += 1

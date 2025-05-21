import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blackhole.tokenizer import tokenize, detokenize, summarize_tokens
from blackhole.embedding import TokenEmbedding, NumberEmbedding, number_embedding_features, prepare_inputs

text = """
During the trial, the compound concentration peaked at 1.23456789e+10 mol/L, but dropped to nearly zero (0.000000000045) within 3.14 seconds. 
Patient survival improved by 123456789%, while the control group had an incidence rate of -0.0000001. 
The total cost exceeded $9,876,543.21, yet the efficiency was measured at 0.0000000000001 units per cycle. 
Unexpectedly, a sample showed a negative reading of -987654321, which contradicted the baseline of 0.0000000001.
"""


# Tokenizacja
tokens, number_map = tokenize(text)
print("Tokens:", tokens)
print("\nNumber Map:")
for idx, (val, typ, raw) in number_map.items():
    print(f"{idx}: {val} ({typ}) -> '{raw}'")

# Przygotowanie inputów
token_ids, num_map, vocab = prepare_inputs(tokens, number_map)

# Inicjalizacja oddzielnych modeli embeddingowych
token_emb_model = TokenEmbedding(vocab_size=len(vocab))
number_emb_model = NumberEmbedding()

# Embedding tokenów
token_embeddings = token_emb_model(token_ids)  # [1, L, token_emb_dim]

# Przygotowanie tensoru cech liczbowych
B, L = token_ids.shape
raw_feats = torch.zeros(B, L, 12)
for b in range(B):
    for i in range(L):
        if i in num_map:
            val, typ, raw = num_map[i]
            raw_feats[b, i] = number_embedding_features(val, typ)

# Embedding liczb
number_embeddings = number_emb_model(raw_feats)  # [1, L, number_emb_dim]

print("\n--- Embeddingi tokenów liczbowych (<num>) ---")
for i in range(len(tokens)):
    token = tokens[i]
    if i in num_map:  # token jest liczbą
        token_emb = token_embeddings[0, i]
        number_emb = number_embeddings[0, i]
        print(f"Token index {i}: '{token}'")
        print(f"Token embedding: {token_emb}")
        print(f"Number embedding: {number_emb}")
        print("-" * 50)

print("\n--- Embeddingi tokenów nieliczbowych (ignorujemy spacje i '<|space|>') ---")
count = 0
max_print = 50
i = 0

while count < max_print and i < len(tokens):
    token = tokens[i]
    if i in num_map or token.strip() == "" or token == "<|space|>":
        i += 1
        continue  # pomijamy liczby (już pokazaliśmy) i spacje
    
    token_emb = token_embeddings[0, i]
    print(f"Token index {i}: '{token}'")
    print(f"Token embedding: {token_emb}")
    print("Number embedding: brak (token nie jest liczbą)")
    print("-" * 50)

    count += 1
    i += 1

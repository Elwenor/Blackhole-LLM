import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blackhole.tokenizer import *
from blackhole.embedding import *

def main():
    text = """
Add 3.14 and -42 to zero. The total sum should be 0.0.  
In physics, the speed of light is approximately 299792458 m/s.  
Negative temperatures like -273.15°C represent absolute zero.
"""

    tokens, number_map = tokenize(text)

    print("Text:\n", text)
    print("\nTokens:", tokens)
    print("\nNumber Map:")
    for idx, (val, typ, raw) in number_map.items():
        print(f"  {idx}: {val} ({typ}) -> '{raw}'")

    token_ids, feats, vocab = prepare_inputs(tokens, number_map)

    print("\nToken IDs:", token_ids)
    print("\nFeatures shape:", feats.shape)

    print("\n--- Numeric Token Embeddings and Decoded Values ---\n")
    print(f"{'Idx':<5} | {'Token':<15} | {'Raw Number':<20} | {'Decoded Value':<20}")
    print("-" * 70)
    
    for idx in sorted(number_map.keys()):
        token = tokens[idx]
        val, typ, raw = number_map[idx]
        feat = feats[0, idx]
        decoded_val = decode_number_from_features(feat)
        print(f"{idx:<5} | {token:<15} | {raw:<20} | {decoded_val:<20}")

if __name__ == "__main__":
    main()

    try:
        test_embedding_decode()
        test_prepare_inputs()
    except NameError:
        print("Test functions are not defined or imported.")

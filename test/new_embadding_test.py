import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blackhole.tokenizer import *
from blackhole.embedding import *

if __name__ == "__main__":
    text = "Add 3.14 and -42 to zero"
    tokens, number_map = tokenize(text)

    print("Text: " + text)

    print("Tokens:", tokens)
    print("Numbers:", number_map)

    token_ids, feats, vocab = prepare_inputs(tokens, number_map)
    print("Tokens ID:", token_ids)
    print("Features shape:", feats.shape)
    print("Features sample (index 2):", feats[0, 2])
    print("Features sample (index 4):", feats[0, 4])

sample_feat_2 = feats[0, 2]
sample_feat_4 = feats[0, 4]

print("Decoded value at idx 2:", decode_number_from_features(sample_feat_2))
print("Original value at idx 2:", 3.14)

print("Decoded value at idx 4:", decode_number_from_features(sample_feat_4))
print("Original value at idx 4:", -42)


test_embedding_decode()
    

test_prepare_inputs()

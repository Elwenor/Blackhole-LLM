import sys
import os
import torch
import math

# Adjust the path to allow importing from the blackhole package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the new, structured path
from blackhole.tokenizer.tokenizer import tokenize
from blackhole.embedding.embedding import number_embedding_features, decode_number_from_features, TokenEmbedding, NumberEmbedding

# NOTE: prepare_inputs was not imported in the original files but is used.
# If prepare_inputs is a utility function that should be in utils/data_prep.py,
# then we should import it from there. For the purpose of this test file,
# I've included a simple version of prepare_inputs here, but in a real project,
# it should be imported.

def prepare_inputs(tokens, number_map, dim=128):
    """
    A simple version of prepare_inputs for testing purposes.
    In a real project, it should be imported from blackhole.utils.data_prep.
    """
    vocab = {token: i for i, token in enumerate(sorted(list(set(tokens))))}
    vocab['<|pad|>'] = len(vocab)
    vocab['<|bos|>'] = len(vocab)
    vocab['<|eos|>'] = len(vocab)
    vocab['<|num|>'] = vocab.get('<|num|>', len(vocab)) # Ensure <|num|> has an ID
    vocab['<|unk|>'] = vocab.get('<|unk|>', len(vocab)) # Ensure <|unk|> has an ID

    token_ids = torch.tensor([vocab.get(token, vocab['<|unk|>']) for token in tokens], dtype=torch.long)
    
    # Initialize an empty numerical features matrix
    # We use -2.0 as the value for missing numerical features, consistent with your evaluation code.
    numeric_features_tensor = torch.full((len(tokens), dim), -2.0, dtype=torch.float32)

    for idx, (val, typ, raw) in number_map.items():
        if idx < len(tokens): # Ensure the index is within bounds
            features = number_embedding_features(val, dim=dim)
            numeric_features_tensor[idx] = torch.tensor(features, dtype=torch.float32)

    return token_ids.unsqueeze(0), numeric_features_tensor.unsqueeze(0), vocab


def run_embedding_demo_and_test():
    print("--- Starting Numerical Embedding Demo and Tests ---")
    print("\n--------------------------------------------------------------")
    print("DEMO 1: Comprehensive Tokenization, Embedding, and Decoding Test")
    print("--------------------------------------------------------------")

    text_complex = """
    During the trial, the compound concentration peaked at 1.23456789e+10 mol/L, but dropped to nearly zero (0.000000000045) within 3.14 seconds.
    Patient survival improved by 123456789%, while the control group had an incidence rate of -0.0000001.
    The total cost exceeded $9,876,543.21, yet the efficiency was measured at 0.0000000000001 units per cycle.
    Unexpectedly, a sample showed a negative reading of -987654321, which contradicted the baseline of 0.0000000001.
    """

    print("\nInput Text (Complex):")
    print(text_complex)

    # Tokenize input text
    tokens_complex, number_map_complex = tokenize(text_complex)
    print("\nTokens:", tokens_complex)
    print("\nNumber Map (original values):")
    for idx, (val, typ, raw) in number_map_complex.items():
        print(f"  Index {idx}: Value={val}, Type='{typ}', Raw='{raw}'")

    # Prepare inputs
    embedding_dim = 128 # Ensure dim is sufficiently large
    token_ids_complex, numeric_features_tensor_complex, vocab_complex = prepare_inputs(tokens_complex, number_map_complex, dim=embedding_dim)

    # Initialize embedding models
    token_emb_model_complex = TokenEmbedding(vocab_size=len(vocab_complex), embedding_dim=embedding_dim)
    number_emb_model_complex = NumberEmbedding(input_dim=embedding_dim, output_dim=embedding_dim)

    # Embed tokens
    token_embeddings_complex = token_emb_model_complex(token_ids_complex)
    number_embeddings_complex = number_emb_model_complex(numeric_features_tensor_complex)

    B, L = token_ids_complex.shape

    print("\n--- Numerical Token Embeddings (<|num|>) and Decoding ---")
    for i in range(L):
        token = tokens_complex[i]
        if i in number_map_complex: # Check if the token at this index is <|num|>
            original_val, _, _ = number_map_complex[i]
            
            current_token_emb = token_embeddings_complex[0, i]
            current_number_feat = numeric_features_tensor_complex[0, i]
            current_number_emb = number_embeddings_complex[0, i]

            decoded_val = decode_number_from_features(current_number_feat.cpu().numpy())

            print(f"Token index {i}: '{token}' (Original Value: {original_val})")
            print(f"  Decoded value from features: {decoded_val}")
            print(f"  Token embedding (from TokenEmbedding): {current_token_emb.detach().cpu().numpy()[:5]}...")
            print(f"  Number embedding (from NumberEmbedding): {current_number_emb.detach().cpu().numpy()[:5]}...")
            print("-" * 50)

    print("\n--- Non-numerical Token Embeddings (excluding spaces and <|num|>) ---")
    count = 0
    max_print_non_numeric = 5 # Limit the number of examples printed
    i = 0
    while count < max_print_non_numeric and i < L:
        token = tokens_complex[i]
        if i in number_map_complex or token.strip() == "":
            i += 1
            continue
        current_token_emb = token_embeddings_complex[0, i]
        print(f"Token index {i}: '{token}'")
        print(f"  Token embedding: {current_token_emb.detach().cpu().numpy()[:5]}...")
        print("  Number embedding: N/A (token is not numerical)")
        print("-" * 50)
        count += 1
        i += 1


    print("\n\n--------------------------------------------------------------")
    print("TEST 2: Numerical Value Decoding Accuracy")
    print("--------------------------------------------------------------")

    text_simple = """
    Add 3.14 and -42 to zero. The total sum should be 0.0. 
    In physics, the speed of light is approximately 299792458 m/s. 
    Negative temperatures like -273.15°C represent absolute zero.
    """

    print("\nInput Text (Simple):")
    print(text_simple)

    tokens_simple, number_map_simple = tokenize(text_simple)

    print("\nTokens:", tokens_simple)
    print("\nNumber Map (original values):")
    for idx, (val, typ, raw) in number_map_simple.items():
        print(f"  {idx}: {val} ({typ}) -> '{raw}'")

    token_ids_simple, feats_simple, vocab_simple = prepare_inputs(tokens_simple, number_map_simple)

    print("\n--- Numerical Value Decoding Verification ---")
    print(f"{'Idx':<5} | {'Token':<15} | {'Original Value':<20} | {'Decoded Value':<20} | {'Status':<10}")
    print("-" * 85)
    
    all_numeric_decoded_correctly = True
    for idx in sorted(number_map_simple.keys()):
        token = tokens_simple[idx]
        original_val, typ, raw = number_map_simple[idx]
        feat = feats_simple[0, idx]
        decoded_val = decode_number_from_features(feat.cpu().numpy())
        
        is_close = math.isclose(original_val, decoded_val, rel_tol=1e-5, abs_tol=1e-8)
        status = "OK" if is_close else "ERROR"
        if not is_close:
            all_numeric_decoded_correctly = False

        print(f"{idx:<5} | {token:<15} | {original_val:<20.10f} | {decoded_val:<20.10f} | {status:<10}")
    
    if all_numeric_decoded_correctly:
        print("\n--> Success: All numerical values were decoded correctly!")
    else:
        print("\n--> WARNING: Some numerical values were NOT decoded correctly.")

    print("\n--- End of Numerical Embedding Demo and Tests ---")


if __name__ == "__main__":
    run_embedding_demo_and_test()
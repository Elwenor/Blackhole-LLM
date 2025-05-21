import torch
import torch.nn as nn
import math
import struct

def number_embedding_features(val: float, typ: str, dim: int = 128) -> torch.Tensor:
    x = float(val)
    x_abs = abs(x)
    x_sign = math.copysign(1.0, x) if x != 0 else 0.0
    
    # Log10 of absolute value, capped for bucket assignment
    log_abs = math.log10(x_abs + 1e-12) if x_abs > 0 else -12.0
    bucket = min(max(int(math.floor(log_abs + 9)), 0), 18)  # 19 buckets total
    
    # Binary encoding of the float64 value (64 bits)
    binary = struct.pack('>d', x)
    bits = ''.join(format(byte, '08b') for byte in binary)
    binary_features = [1.0 if b == '1' else -1.0 for b in bits]  # -1/1 encoding
    
    # Semantic features to help model generalize
    semantic_features = [
        x_sign,                              # sign (+1/-1/0)
        1.0 if x == 0 else -1.0,             # is_zero feature
        1.0 if typ == 'int' else -1.0,       # is_int feature
        1.0 if typ == 'float' else -1.0,     # is_float feature
        math.tanh(x),                        # bounded raw value
        math.tanh(log_abs),                  # bounded log magnitude
    ]
    
    # Bucket one-hot with -1/1 encoding for better norm
    bucket_features = [-1.0] * 19
    bucket_features[bucket] = 1.0
    
    # Combine all features
    features = semantic_features + bucket_features + binary_features
    
    # Pad or truncate to dim
    if len(features) < dim:
        features += [-1.0] * (dim - len(features))
    else:
        features = features[:dim]
    
    return torch.tensor(features, dtype=torch.float)

def decode_number_from_features(features: torch.Tensor) -> float:
    # Minimum length needed for full decoding (semantic + bucket + binary)
    min_len_for_full_decode = 6 + 19 + 64  # 89

    if len(features) >= min_len_for_full_decode:
        # Full binary decode
        binary_features = features[6+19:6+19+64].tolist()
        binary_str = ''.join('1' if feat > 0 else '0' for feat in binary_features)
        binary_bytes = bytearray()
        for i in range(0, len(binary_str), 8):
            if i + 8 <= len(binary_str):
                byte = int(binary_str[i:i+8], 2)
                binary_bytes.append(byte)
        if len(binary_bytes) == 8:
            try:
                return struct.unpack('>d', bytes(binary_bytes))[0]
            except struct.error:
                pass  # Fallback below

    # Fallback: if tensor is too short or decode failed
    if len(features) < 6:
        raise ValueError("Features tensor too small for decoding.")

    x_sign = features[0].item()
    is_zero = features[1].item() > 0
    if is_zero:
        return 0.0

    tanh_x = features[4].item()
    # atanh for bounded values, with protection
    try:
        x_approx = math.atanh(tanh_x) if abs(tanh_x) < 0.99999 else math.copysign(100.0, tanh_x)
    except Exception:
        x_approx = 0.0

    return x_approx if x_sign >= 0 else -x_approx


class NumberEmbedding(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


def prepare_inputs(tokens: list, number_map: dict, feat_dim: int = 128):
    vocab = {tok: idx for idx, tok in enumerate(sorted(set(tokens)))}
    token_ids = [vocab[t] for t in tokens]
    token_ids_tensor = torch.LongTensor([token_ids])  # [1, L]

    seq_len = len(tokens)
    feats = torch.ones((1, seq_len, feat_dim), dtype=torch.float) * -1.0  # initialize with -1

    for idx, data in number_map.items():
        if idx < 0 or idx >= seq_len:
            print(f"[WARN] Ignoring number_map index {idx} out of bounds (0-{seq_len-1})")
            continue
            
        # Handle both 2-element and 3-element tuples
        if isinstance(data, (tuple, list)):
            if len(data) >= 2:  # Extract just the first two elements (value, type)
                val, typ = data[0], data[1]
                feats[0, idx] = number_embedding_features(val, typ, dim=feat_dim)
            else:
                print(f"[WARN] Malformed number_map entry at index {idx}: {data} - not enough elements")
        else:
            print(f"[WARN] Malformed number_map entry at index {idx}: {data} - not a tuple/list")

    return token_ids_tensor, feats, vocab


def test_embedding_decode():
    test_values = [
        (3.14, 'float'),
        (-42, 'int'),
        (1234567890.123456, 'float'),
        (-9876543210, 'int'),
        (1e-10, 'float'),
        (1e20, 'float'),
        (0, 'int')
    ]
    
    for val, typ in test_values:
        feats = number_embedding_features(val, typ, dim=128)
        decoded = decode_number_from_features(feats)
        rel_err = abs(decoded - val) / max(abs(val), 1e-10) if val != 0 else abs(decoded)
        print(f"Original: {val}, Decoded: {decoded}, Rel. error: {rel_err:.8f}")

def test_prepare_inputs():
    tokens = ['<|cap|>', 'add', '<|num|>', 'and', '<|num|>', 'to', '<|space|>', 'zero']
    
    # Test with standard 2-element tuple format
    number_map_correct = {
        2: (3.14, 'float'),
        4: (-42, 'int')
    }
    
    print("\nTesting with correct number_map format:")
    token_ids, feats, vocab = prepare_inputs(tokens, number_map_correct)
    print(f"Tokens ID: {token_ids}")
    print(f"Features shape: {feats.shape}")
    
    # Decode and verify numbers
    print("\nVerifying number decoding:")
    for idx in number_map_correct:
        original_val = number_map_correct[idx][0]
        decoded_val = decode_number_from_features(feats[0, idx])
        print(f"Position {idx}: Original = {original_val}, Decoded = {decoded_val}")
    
    # Test with 3-element tuple format (compatibility with your current code)
    number_map_with_text = {
        2: (3.14, 'float', ' 3.14'),
        4: (-42, 'int', ' -42')
    }
    
    print("\nTesting with 3-element tuple format:")
    token_ids, feats, vocab = prepare_inputs(tokens, number_map_with_text)
    
    # Decode and verify numbers
    print("Verifying number decoding with 3-element format:")
    for idx in number_map_with_text:
        original_val = number_map_with_text[idx][0]
        decoded_val = decode_number_from_features(feats[0, idx])
        print(f"Position {idx}: Original = {original_val}, Decoded = {decoded_val}")

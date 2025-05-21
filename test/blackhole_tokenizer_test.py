import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blackhole.tokenizer import tokenize, detokenize, summarize_tokens

text = """
The price rose from $1,234.56 on 2023-05-20 to 0x1A3F units by 12:30 PM. Meanwhile, the experimental drug reduced the virus count by 0.000123 units...
"""

tokens, number_map = tokenize(text)

print("Tokens:", tokens)

print("\nNumber Map (token index → (value, type, raw)):")
for idx, (val, typ, raw) in number_map.items():
    print(f"{idx}: {val} ({typ}), raw: {raw}")

print("\nToken Summary:")
for tok, idx, count in summarize_tokens(tokens):
    print(f"ID: {idx:2d} | Token: '{tok}' | Count: {count}")

unique_tokens = len(set(tokens))
print(f"\nNumber of unique tokens: {unique_tokens}")

print("Orgianlny text: " + text)

print("\nDetokenized with numbers:")
print(detokenize(tokens, number_map))

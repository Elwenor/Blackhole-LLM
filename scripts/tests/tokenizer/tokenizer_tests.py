import sys
import os
import torch # Not strictly needed for tokenizer alone, but often present in PyTorch projects
from transformers import GPT2TokenizerFast, BertTokenizerFast
from datasets import load_dataset # Make sure you have datasets installed: pip install datasets

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..\..')))

from blackhole.tokenizer import *

DATASETS = [
    {"name": "General Text", "dataset": "wikitext", "subset": "wikitext-2-raw-v1", "split": "train", "field": "text"},
    {"name": "Scientific Text", "dataset": "ccdv/arxiv-summarization", "subset": None, "split": "train", "field": "article"},
    {"name": "Mathematical Text", "dataset": "JeanKaddour/minipile", "subset": None, "split": "train", "field": "text",
     "filter_func": lambda x: any(w in x["text"].lower() for w in ["theorem", "equation", "math"])}
]

MAX_CHARS = 2000  # Limit text length to avoid loading entire books

def run_basic_tokenizer_test():
    """
    Runs a basic sanity check on the tokenizer's core functions.
    Content derived from blackhole_tokenizer_test.py.
    """
    print("\n--------------------------------------------------------------")
    print("TEST 1: Basic Tokenizer Functionality Check")
    print("--------------------------------------------------------------")

    text = """
The price rose from $1,234.56 on 2023-05-20 to 0x1A3F units by 12:30 PM. Meanwhile, the experimental drug reduced the virus count by 0.000123 units...
"""
    print(f"\nOriginal Text:\n{text}")

    tokens, number_map = tokenize(text)

    print("\nTokens:", tokens)

    print("\nNumber Map (token index → (value, type, raw)):")
    for idx, (val, typ, raw) in number_map.items():
        print(f"{idx}: {val} ({typ}), raw: {raw}")

    print("\nToken Summary:")
    for tok, idx, count in summarize_tokens(tokens):
        print(f"ID: {idx:2d} | Token: '{tok}' | Count: {count}")

    unique_tokens = len(set(tokens))
    print(f"\nNumber of unique tokens: {unique_tokens}")

    detok_text = detokenize(tokens, number_map)
    print("\nDetokenized Text (with numbers):")
    print(detok_text)
    print(f"Detokenization Exact Match with Original: {detok_text == text.strip()}") # strip for leading/trailing whitespace


def run_tokenizer_comparison_demo(num_samples=3):
    """
    Compares Blackhole tokenizer's detokenization with GPT-2 and BERT
    on various real-world datasets.
    Content derived from tokenizer_example.py.
    """
    print("\n--------------------------------------------------------------")
    print("DEMO 2: Blackhole Tokenizer Comparison with Hugging Face Models")
    print("--------------------------------------------------------------")

    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    for ds_cfg in DATASETS:
        print(f"\n=== Dataset: {ds_cfg['name']} ===")
        ds = load_dataset(ds_cfg["dataset"], ds_cfg.get("subset"), split=ds_cfg["split"])

        if ds_cfg.get("filter_func"):
            ds = ds.filter(ds_cfg["filter_func"])

        count = 0
        for sample in ds:
            text = sample[ds_cfg["field"]]
            # Filter out empty or too short texts from datasets
            if not text or len(text.strip()) < 50:
                continue

            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "..."

            print(f"\n--- Sample {count+1} (len={len(text)}) ---")
            print("Original:")
            print(text.replace("\n", " ").strip()) # Clean up newlines for display

            # Blackhole Tokenizer
            bh_tokens, number_map = tokenize(text)
            bh_detok = detokenize(bh_tokens, number_map)
            # Normalize whitespace for comparison
            normalized_original = ' '.join(text.split()).strip()
            normalized_bh_detok = ' '.join(bh_detok.split()).strip()

            print("\nBlackhole Tokenization:")
            print(f"  Detokenization Exact Match (normalized): {normalized_bh_detok == normalized_original}")
            print("  Detokenized snippet:")
            print(bh_detok[:500].replace("\n", " ").strip() + ("..." if len(bh_detok) > 500 else ""))

            # GPT-2 Tokenizer
            gpt2_tokens = gpt2_tok.encode(text)
            gpt2_detok = gpt2_tok.decode(gpt2_tokens)
            normalized_gpt2_detok = ' '.join(gpt2_detok.split()).strip()

            print("\nGPT-2 Tokenization:")
            print(f"  Detokenization Exact Match (normalized): {normalized_gpt2_detok == normalized_original}")
            print("  Detokenized snippet:")
            print(gpt2_detok[:500].replace("\n", " ").strip() + ("..." if len(gpt2_detok) > 500 else ""))

            # BERT Tokenizer
            bert_tokens = bert_tok.encode(text, add_special_tokens=False)
            bert_detok = bert_tok.decode(bert_tokens)
            normalized_bert_detok = ' '.join(bert_detok.split()).strip()

            print("\nBERT Tokenization:")
            print(f"  Detokenization Exact Match (normalized): {normalized_bert_detok == normalized_original}")
            print("  Detokenized snippet:")
            print(bert_detok[:500].replace("\n", " ").strip() + ("..." if len(bert_detok) > 500 else ""))

            count += 1
            if count >= num_samples:
                break


if __name__ == "__main__":
    run_basic_tokenizer_test()
    run_tokenizer_comparison_demo()
    print("\n--- End of Tokenizer Tests and Demo ---")
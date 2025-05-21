from transformers import GPT2TokenizerFast, BertTokenizerFast
from datasets import load_dataset
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blackhole.tokenizer import tokenize, detokenize

DATASETS = [
    {"name": "General Text", "dataset": "wikitext", "subset": "wikitext-2-raw-v1", "split": "train", "field": "text"},
    {"name": "Scientific Text", "dataset": "ccdv/arxiv-summarization", "subset": None, "split": "train", "field": "article"},
    {"name": "Mathematical Text", "dataset": "JeanKaddour/minipile", "subset": None, "split": "train", "field": "text",
     "filter_func": lambda x: any(w in x["text"].lower() for w in ["theorem", "equation", "math"])}
]

MAX_CHARS = 2000  # Limit długości tekstu, żeby nie ładować całych ksiąg

def sample_and_show_detokenization(num_samples=3):
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

            if len(text) > MAX_CHARS:
                # Opcjonalnie możemy pociąć tekst, albo pominąć
                text = text[:MAX_CHARS] + "..."

            print(f"\n--- Sample {count+1} (len={len(text)}) ---")
            print("Original:")
            print(text.replace("\n", " "))

            # Blackhole
            bh_tokens, number_map = tokenize(text)
            bh_detok = detokenize(bh_tokens, number_map)
            print("\nBlackhole Detokenization Exact:", bh_detok == text)
            print("Blackhole Detokenization snippet:")
            print(bh_detok[:500].replace("\n", " ") + ("..." if len(bh_detok) > 500 else ""))

            # GPT-2
            gpt2_tokens = gpt2_tok.encode(text)
            gpt2_detok = gpt2_tok.decode(gpt2_tokens)
            print("\nGPT-2 Detokenization Exact:", gpt2_detok == text)
            print("GPT-2 Detokenization snippet:")
            print(gpt2_detok[:500].replace("\n", " ") + ("..." if len(gpt2_detok) > 500 else ""))

            # BERT
            bert_tokens = bert_tok.encode(text, add_special_tokens=False)
            bert_detok = bert_tok.decode(bert_tokens)
            print("\nBERT Detokenization Exact:", bert_detok == text)
            print("BERT Detokenization snippet:")
            print(bert_detok[:500].replace("\n", " ") + ("..." if len(bert_detok) > 500 else ""))

            count += 1
            if count >= num_samples:
                break

if __name__ == "__main__":
    sample_and_show_detokenization()

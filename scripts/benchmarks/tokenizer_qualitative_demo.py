from transformers import GPT2TokenizerFast, BertTokenizerFast
from datasets import load_dataset
import sys, os
import re # Make sure re is imported here as well if not already done

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

from blackhole.tokenizer.tokenizer import *

DATASETS = [
    {
        "name": "Scientific Text",
        "dataset": "ccdv/arxiv-summarization",
        "subset": None,
        "split": "train",
        "text_field": "article"
    },
    {
        "name": "Mathematical Text",
        "dataset": "JeanKaddour/minipile",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "filter_func": lambda example: any(word in example["text"].lower() for word in ["theorem", "equation", "math"])
    }
]

def print_summary(name, tokens, detok, number_map=None, show_map=False, original_text=None):
    print(f"\n=== {name} ===")
    if original_text is not None:
        print("Original text:")
        print(original_text)
        print("-" * 40)
    print("Detokenized text:")
    print(detok)
    print(f"Total tokens: {len(tokens)}")
    print(f"Unique tokens: {len(set(tokens))}")
    if show_map and number_map:
        print("\nNumber Map (token idx → (value, type, raw)):")
        for idx, (val, typ, raw) in number_map.items():
            print(f"{idx}: {val} ({typ}), raw: {raw}")


def get_sample_text(config):
    print(f"Loading sample from dataset: {config['name']}")
    ds = load_dataset(config["dataset"], config.get("subset"), split=config["split"])

    if config.get("filter_func"):
        ds = ds.filter(config["filter_func"])
        if len(ds) == 0:
            raise ValueError(f"No samples found for filter in {config['name']} dataset")

    text = ds[0][config["text_field"]]

    # Limit tekstu do 1500 znaków
    if len(text) > 1500:
        text = text[:1500] + "... [truncated]\n"

    return text


def main():
    # Load tokenizers once
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
    bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    for ds_conf in DATASETS:
        text = get_sample_text(ds_conf)

        # Blackhole tokenize/detokenize
        bh_tokens, number_map = tokenize(text)
        bh_detok = detokenize(bh_tokens, number_map)
        print_summary(f"Blackhole ({ds_conf['name']})", bh_tokens, bh_detok, number_map, show_map=True, original_text=text)

        # GPT-2 tokenize/detokenize
        gpt2_tokens = gpt2_tok.tokenize(text)
        gpt2_detok = gpt2_tok.convert_tokens_to_string(gpt2_tokens)
        print_summary(f"GPT-2 ({ds_conf['name']})", gpt2_tokens, gpt2_detok, original_text=text)

        # BERT tokenize/detokenize
        bert_tokens = bert_tok.tokenize(text)
        bert_detok = bert_tok.convert_tokens_to_string(bert_tokens)
        print_summary(f"BERT ({ds_conf['name']})", bert_tokens, bert_detok, original_text=text)


if __name__ == "__main__":
    main()
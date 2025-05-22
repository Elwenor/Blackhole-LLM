import sys
import os
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from tabulate import tabulate
from transformers import GPT2TokenizerFast, BertTokenizerFast
import re # Import regex for normalization

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from blackhole.tokenizer_hugging_face import BlackholeTokenizer
except ImportError as e:
    print(f"Error importing BlackholeTokenizer: {e}")
    print("Please ensure your BlackholeTokenizer is correctly located and importable.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


MAX_TOKENS = 10_000_000 

DATASETS = [
    {
        "name": "General Text",
        "dataset": "wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "train",
        "text_field": "text"
    },
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

def normalize_text(text):
    """
    Normalizes text by:
    - Stripping leading/trailing whitespace
    - Replacing multiple spaces with a single space
    - Lowercasing (important for BERT comparison)
    - Removing specific Blackhole special tokens if present (for a fairer match)
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
    # Specific to Blackhole if it adds visible tokens like [CAP], [NUM], [ALLCAPS]
    # You might need to adjust this based on the actual special tokens Blackhole uses.
    text = re.sub(r'\[CAP\]|\[NUM\]|\[ALLCAPS\]', '', text) 
    return text.lower() # Lowercase for BERT's typical behavior

def load_data_samples(config, num_samples=25):
    """
    Load dataset samples, optionally filter, and truncate to approximate MAX_TOKENS.
    Rough heuristic: average 4 chars per token.
    """
    print(f"Loading dataset: {config['name']}")
    ds = load_dataset(config["dataset"], config.get("subset"), split=config["split"])

    if config.get("filter_func"):
        ds = ds.filter(config["filter_func"])

    total_chars_estimate = 0
    texts = []

    # Using min to ensure we don't request more samples than available in a small dataset
    actual_num_samples = min(num_samples, len(ds))
    for idx in range(actual_num_samples):
        text = ds[idx][config["text_field"]]
        if text.strip(): # Only include non-empty texts for meaningful benchmarking
            texts.append(text)
            total_chars_estimate += len(text)
            # Stop if we have enough data for the rough token estimate
            if total_chars_estimate / 4 >= MAX_TOKENS:
                break

    return texts

def run_benchmark(texts, dataset_name):
    """
    Run tokenization and detokenization benchmarks on the provided texts.
    Return list of dicts with all results per sample.
    """
    results = []

    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
    bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    bh_tok = BlackholeTokenizer()
    print(f"\n--- Training BlackholeTokenizer for {dataset_name} ---")
    train_start_time = time.time()
    # Ensure train_tokenizer is called with an iterable that yields strings
    bh_tok.train_tokenizer(iter(texts), vocab_size=30000, min_freq=2, show_progress=True) 
    train_end_time = time.time()
    print(f"--- BlackholeTokenizer trained in {train_end_time - train_start_time:.2f} seconds. ---")

    output_dir = f"./blackhole_tokenizer_benchmark_trained_{dataset_name.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    bh_tok.save_pretrained(output_dir)
    loaded_bh_tok = BlackholeTokenizer.from_pretrained(output_dir)
    print(f"BlackholeTokenizer for {dataset_name} loaded from {output_dir}")

    for idx, text in enumerate(texts, 1):
        # Truncate original text for example printing
        original_text_example = text[:300] + "..." if len(text) > 300 else text

        print(f"Processing sample {idx}/{len(texts)} from {dataset_name} ({len(text)} chars)")

        sample_res = {"dataset": dataset_name, "sample_id": idx, "text_length": len(text)}
        sample_res["original_text_example"] = original_text_example

        # Blackhole tokenize/detokenize
        start = time.time()
        # For Blackhole, `add_special_tokens=True` might add tokens like [CLS], [SEP] but also
        # internal special tokens like [CAP], [NUM] that Blackhole handles during detokenization
        # for semantic meaning. We need to tell decode to skip the ones it adds for internal use
        # if we want exact string match for 'Exact Match' metric.
        bh_enc = loaded_bh_tok(text, add_special_tokens=True, truncation=True, max_length=512, return_tensors=None)
        bh_tokens_ids = bh_enc["input_ids"]
        sample_res["blackhole_time_ms"] = (time.time() - start) * 1000

        start = time.time()
        # Crucial change: skip_special_tokens=True for Blackhole to remove [CLS], [SEP] and
        # any other 'added' special tokens.
        # However, Blackhole's internal [CAP], [NUM] might still be present if they are part of its vocabulary
        # and not filtered by `skip_special_tokens`. If so, we'll need `normalize_text` to handle them.
        bh_detok = loaded_bh_tok.decode(bh_tokens_ids, skip_special_tokens=True) 
        sample_res["blackhole_detok_time_ms"] = (time.time() - start) * 1000
        sample_res["blackhole_detok_example"] = bh_detok[:300] + "..." if len(bh_detok) > 300 else bh_detok

        numbers_detected = 0
        if loaded_bh_tok._last_original_metadata_for_decode:
            if len(loaded_bh_tok._last_original_metadata_for_decode) > 0:
                metadata_list_for_current_seq = loaded_bh_tok._last_original_metadata_for_decode[0][0]
                numbers_detected = sum(1 for m in metadata_list_for_current_seq if m.get('type') == 'NUM')

        # Use normalized text for exact match comparison
        sample_res.update({
            "blackhole_tokens": len(bh_tokens_ids),
            "blackhole_unique_tokens": len(set(bh_tokens_ids)),
            "blackhole_exact_match": normalize_text(bh_detok) == normalize_text(text),
            "blackhole_numbers_detected": numbers_detected
        })

        # GPT-2 tokenize/detokenize
        start = time.time()
        gpt2_tokens = gpt2_tok.encode(text)
        sample_res["gpt2_time_ms"] = (time.time() - start) * 1000

        start = time.time()
        # GPT-2 decode typically handles special tokens well, and usually reconstructs exactly.
        gpt2_detok = gpt2_tok.decode(gpt2_tokens)
        sample_res["gpt2_detok_time_ms"] = (time.time() - start) * 1000
        sample_res["gpt2_detok_example"] = gpt2_detok[:300] + "..." if len(gpt2_detok) > 300 else gpt2_detok

        sample_res.update({
            "gpt2_tokens": len(gpt2_tokens),
            "gpt2_unique_tokens": len(set(gpt2_tokens)),
            "gpt2_exact_match": normalize_text(gpt2_detok) == normalize_text(text) # Use normalized for consistency
        })

        # BERT tokenize/detokenize
        start = time.time()
        # `add_special_tokens=False` here is good as BERT often adds [CLS], [SEP] by default.
        bert_enc = bert_tok.encode_plus(text, add_special_tokens=False, truncation=True, max_length=512)
        bert_tokens = bert_enc["input_ids"]
        sample_res["bert_time_ms"] = (time.time() - start) * 1000

        start = time.time()
        # Crucial change: skip_special_tokens=True for BERT to remove [CLS], [SEP].
        bert_detok = bert_tok.decode(bert_tokens, skip_special_tokens=True) 
        sample_res["bert_detok_time_ms"] = (time.time() - start) * 1000
        sample_res["bert_detok_example"] = bert_detok[:300] + "..." if len(bert_detok) > 300 else bert_detok

        sample_res.update({
            "bert_tokens": len(bert_tokens),
            "bert_unique_tokens": len(set(bert_tokens)),
            "bert_exact_match": normalize_text(bert_detok) == normalize_text(text) # Use normalized for consistency
        })

        # Compute chars per token for each tokenizer
        for key in ["blackhole", "gpt2", "bert"]:
            token_count = sample_res.get(f"{key}_tokens", 0)
            sample_res[f"{key}_chars_per_token"] = (len(text) / token_count) if token_count > 0 else 0


        results.append(sample_res)

    return results

def generate_summary_table(results):
    df = pd.DataFrame(results)

    summary = df.groupby('dataset').agg({
        'text_length': 'mean',
        'blackhole_tokens': 'mean',
        'blackhole_unique_tokens': 'mean',
        'blackhole_time_ms': 'mean',
        'blackhole_detok_time_ms': 'mean',
        'blackhole_exact_match': 'mean',
        'blackhole_numbers_detected': 'mean',
        'blackhole_chars_per_token': 'mean',
        'gpt2_tokens': 'mean',
        'gpt2_unique_tokens': 'mean',
        'gpt2_time_ms': 'mean',
        'gpt2_detok_time_ms': 'mean',
        'gpt2_exact_match': 'mean',
        'gpt2_chars_per_token': 'mean',
        'bert_tokens': 'mean',
        'bert_unique_tokens': 'mean',
        'bert_time_ms': 'mean',
        'bert_detok_time_ms': 'mean',
        'bert_exact_match': 'mean',
        'bert_chars_per_token': 'mean'
    }).reset_index()

    for idx, row in summary.iterrows():
        summary.at[idx, 'bh_vs_gpt2_tokens'] = row['blackhole_tokens'] / row['gpt2_tokens'] if row['gpt2_tokens']>0 else float('inf')
        summary.at[idx, 'bh_vs_bert_tokens'] = row['blackhole_tokens'] / row['bert_tokens'] if row['bert_tokens']>0 else float('inf')
        summary.at[idx, 'bh_vs_gpt2_speed']  = row['blackhole_time_ms'] / row['gpt2_time_ms'] if row['gpt2_time_ms']>0 else float('inf')
        summary.at[idx, 'bh_vs_bert_speed']  = row['blackhole_time_ms'] / row['bert_time_ms'] if row['bert_time_ms']>0 else float('inf')

    overall = df.mean(numeric_only=True).to_dict()
    overall['dataset'] = 'OVERALL'
    overall['bh_vs_gpt2_tokens'] = overall['blackhole_tokens']/overall['gpt2_tokens'] if overall['gpt2_tokens']>0 else float('inf')
    overall['bh_vs_bert_tokens'] = overall['blackhole_tokens']/overall['bert_tokens'] if overall['bert_tokens']>0 else float('inf')
    overall['bh_vs_gpt2_speed']  = overall['blackhole_time_ms']/overall['gpt2_time_ms'] if overall['gpt2_time_ms']>0 else float('inf')
    overall['bh_vs_bert_speed']  = overall['blackhole_time_ms']/overall['bert_time_ms'] if overall['bert_time_ms']>0 else float('inf')
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

    return summary

def print_summary_tables(summary, all_results):
    formatters = {
        'text_length': '{:.1f}',
        'blackhole_tokens': '{:.1f}',
        'blackhole_unique_tokens': '{:.1f}',
        'blackhole_time_ms': '{:.2f}',
        'blackhole_detok_time_ms': '{:.2f}',
        'blackhole_exact_match': '{:.2%}',
        'blackhole_numbers_detected': '{:.1f}',
        'blackhole_chars_per_token': '{:.2f}',
        'gpt2_tokens': '{:.1f}',
        'gpt2_unique_tokens': '{:.1f}',
        'gpt2_time_ms': '{:.2f}',
        'gpt2_detok_time_ms': '{:.2f}',
        'gpt2_exact_match': '{:.2%}',
        'gpt2_chars_per_token': '{:.2f}',
        'bert_tokens': '{:.1f}',
        'bert_unique_tokens': '{:.1f}',
        'bert_time_ms': '{:.2f}',
        'bert_detok_time_ms': '{:.2f}',
        'bert_exact_match': '{:.2%}',
        'bert_chars_per_token': '{:.2f}',
        'bh_vs_gpt2_tokens': '{:.3f}',
        'bh_vs_bert_tokens': '{:.3f}',
        'bh_vs_gpt2_speed': '{:.3f}',
        'bh_vs_bert_speed': '{:.3f}'
    }
    df = summary.copy()
    for col, fmt in formatters.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt.format(x))

    basic = df[['dataset','text_length','blackhole_tokens','gpt2_tokens','bert_tokens',
                 'blackhole_time_ms','gpt2_time_ms','bert_time_ms']]
    basic.columns = ['Dataset','Avg Text Length','BH Tokens','GPT-2 Tokens','BERT Tokens',
                      'BH Time (ms)','GPT-2 Time (ms)','BERT Time (ms)']
    print("\n== BASIC METRICS ==")
    print(tabulate(basic, headers='keys', tablefmt='grid', showindex=False))
    
    # Example for General Text
    example_general_text = next((r for r in all_results if r['dataset'] == 'General Text'), None)
    if example_general_text:
        print("\n--- Example for General Text ---")
        print(f"Original (first 300 chars): {example_general_text['original_text_example']}")
        print(f"Blackhole Detokenized (first 300 chars): {example_general_text['blackhole_detok_example']}")
        print(f"GPT-2 Detokenized (first 300 chars): {example_general_text['gpt2_detok_example']}")
        print(f"BERT Detokenized (first 300 chars): {example_general_text['bert_detok_example']}")


    adv = df[['dataset','blackhole_chars_per_token','gpt2_chars_per_token','bert_chars_per_token',
              'blackhole_exact_match','gpt2_exact_match','bert_exact_match','blackhole_numbers_detected']]
    adv.columns = ['Dataset','BH Chars/Token','GPT-2 Chars/Token','BERT Chars/Token',
                    'BH Exact Match','GPT-2 Exact Match','BERT Exact Match','Numbers Detected']
    print("\n== ADVANCED METRICS ==")
    print(tabulate(adv, headers='keys', tablefmt='grid', showindex=False))

    # Example for Scientific Text
    example_scientific_text = next((r for r in all_results if r['dataset'] == 'Scientific Text'), None)
    if example_scientific_text:
        print("\n--- Example for Scientific Text ---")
        print(f"Original (first 300 chars): {example_scientific_text['original_text_example']}")
        print(f"Blackhole Detokenized (first 300 chars): {example_scientific_text['blackhole_detok_example']}")
        print(f"GPT-2 Detokenized (first 300 chars): {example_scientific_text['gpt2_detok_example']}")
        print(f"BERT Detokenized (first 300 chars): {example_scientific_text['bert_detok_example']}")


    comp = df[['dataset','bh_vs_gpt2_tokens','bh_vs_bert_tokens','bh_vs_gpt2_speed','bh_vs_bert_speed']]
    comp.columns = ['Dataset','BH/GPT-2 Tokens Ratio','BH/BERT Tokens Ratio',
                     'BH/GPT-2 Speed Ratio','BH/BERT Speed Ratio']
    print("\n== COMPARATIVE METRICS ==")
    print(tabulate(comp, headers='keys', tablefmt='grid', showindex=False))

    # Example for Mathematical Text
    example_mathematical_text = next((r for r in all_results if r['dataset'] == 'Mathematical Text'), None)
    if example_mathematical_text:
        print("\n--- Example for Mathematical Text ---")
        print(f"Original (first 300 chars): {example_mathematical_text['original_text_example']}")
        print(f"Blackhole Detokenized (first 300 chars): {example_mathematical_text['blackhole_detok_example']}")
        print(f"GPT-2 Detokenized (first 300 chars): {example_mathematical_text['gpt2_detok_example']}")
        print(f"BERT Detokenized (first 300 chars): {example_mathematical_text['bert_detok_example']}")


def plot_results(summary):
    datasets = summary['dataset'].values
    bar_width = 0.2
    x = np.arange(len(datasets))

    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.bar(x-bar_width, summary['blackhole_tokens'].astype(float), width=bar_width, label='Blackhole')
    plt.bar(x,            summary['gpt2_tokens'].astype(float), width=bar_width, label='GPT-2')
    plt.bar(x+bar_width, summary['bert_tokens'].astype(float), width=bar_width, label='BERT')
    plt.xticks(x, datasets, rotation=45)
    plt.title("Avg Token Counts"); plt.legend()

    plt.subplot(1,2,2)
    plt.bar(x-bar_width, summary['blackhole_time_ms'].astype(float), width=bar_width, label='Blackhole')
    plt.bar(x,            summary['gpt2_time_ms'].astype(float), width=bar_width, label='GPT-2')
    plt.bar(x+bar_width, summary['bert_time_ms'].astype(float), width=bar_width, label='BERT')
    plt.xticks(x, datasets, rotation=45)
    plt.title("Avg Tokenization Time (ms)"); plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    all_results = []
    for ds_config in DATASETS:
        samples = load_data_samples(ds_config, num_samples=25)
        
        if not samples:
            print(f"Skipping {ds_config['name']} due to no valid samples found.")
            continue

        results = run_benchmark(samples, ds_config["name"])
        all_results.extend(results)

    if all_results: # Only generate summary if there are results
        summary = generate_summary_table(all_results)
        print_summary_tables(summary, all_results) 
        plot_results(summary)
    else:
        print("No benchmark results to display. Check dataset loading and sample availability.")

if __name__ == "__main__":
    main()
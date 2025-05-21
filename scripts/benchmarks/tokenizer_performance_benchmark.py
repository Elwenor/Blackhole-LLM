from transformers import GPT2TokenizerFast, BertTokenizerFast
import sys
import os
import time
import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from tabulate import tabulate

# Ensure blackhole tokenizer is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

from blackhole.tokenizer import *

MAX_TOKENS = 10_000_000  # Max tokens limit across all samples

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

def load_data_samples(config, num_samples=25):
    """
    Load dataset samples, optionally filter, and truncate to approximate MAX_TOKENS.
    Rough heuristic: average 4 chars per token.
    """
    print(f"Loading dataset: {config['name']}")
    ds = load_dataset(config["dataset"], config.get("subset"), split=config["split"])
    
    if config.get("filter_func"):
        ds = ds.filter(config["filter_func"])
    
    total_tokens_estimate = 0
    texts = []
    
    num_samples = min(num_samples, len(ds))
    for idx in range(num_samples):
        text = ds[idx][config["text_field"]]
        if total_tokens_estimate >= MAX_TOKENS:
            break
        texts.append(text)
        total_tokens_estimate += len(text) / 4
    
    return texts

def run_benchmark(texts, dataset_name):
    """
    Run tokenization and detokenization benchmarks on the provided texts.
    Return list of dicts with all results per sample.
    """
    results = []
    
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    for idx, text in enumerate(texts, 1):
        print(f"Processing sample {idx}/{len(texts)} from {dataset_name} ({len(text)} chars)")
        
        sample_res = {"dataset": dataset_name, "sample_id": idx, "text_length": len(text)}
        
        # Blackhole tokenize/detokenize
        start = time.time()
        bh_tokens, number_map = tokenize(text)
        sample_res["blackhole_time_ms"] = (time.time() - start) * 1000
        
        start = time.time()
        bh_detok = detokenize(bh_tokens, number_map)
        sample_res["blackhole_detok_time_ms"] = (time.time() - start) * 1000
        
        sample_res.update({
            "blackhole_tokens": len(bh_tokens),
            "blackhole_unique_tokens": len(set(bh_tokens)),
            "blackhole_exact_match": bh_detok == text,
            "blackhole_numbers_detected": len(number_map) if number_map else 0
        })
        
        # GPT-2 tokenize/detokenize
        start = time.time()
        gpt2_tokens = gpt2_tok.encode(text)
        sample_res["gpt2_time_ms"] = (time.time() - start) * 1000
        
        start = time.time()
        gpt2_detok = gpt2_tok.decode(gpt2_tokens)
        sample_res["gpt2_detok_time_ms"] = (time.time() - start) * 1000
        
        sample_res.update({
            "gpt2_tokens": len(gpt2_tokens),
            "gpt2_unique_tokens": len(set(gpt2_tokens)),
            "gpt2_exact_match": gpt2_detok == text
        })
        
        # BERT tokenize/detokenize
        start = time.time()
        bert_enc = bert_tok.encode_plus(text, add_special_tokens=False)
        bert_tokens = bert_enc["input_ids"]
        sample_res["bert_time_ms"] = (time.time() - start) * 1000
        
        start = time.time()
        bert_detok = bert_tok.decode(bert_tokens)
        sample_res["bert_detok_time_ms"] = (time.time() - start) * 1000
        
        sample_res.update({
            "bert_tokens": len(bert_tokens),
            "bert_unique_tokens": len(set(bert_tokens)),
            "bert_exact_match": bert_detok == text
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

def print_summary_tables(summary):
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
    
    adv = df[['dataset','blackhole_chars_per_token','gpt2_chars_per_token','bert_chars_per_token',
              'blackhole_exact_match','gpt2_exact_match','bert_exact_match','blackhole_numbers_detected']]
    adv.columns = ['Dataset','BH Chars/Token','GPT-2 Chars/Token','BERT Chars/Token',
                   'BH Exact Match','GPT-2 Exact Match','BERT Exact Match','Numbers Detected']
    print("\n== ADVANCED METRICS ==")
    print(tabulate(adv, headers='keys', tablefmt='grid', showindex=False))
    
    comp = df[['dataset','bh_vs_gpt2_tokens','bh_vs_bert_tokens','bh_vs_gpt2_speed','bh_vs_bert_speed']]
    comp.columns = ['Dataset','BH/GPT-2 Tokens Ratio','BH/BERT Tokens Ratio',
                    'BH/GPT-2 Speed Ratio','BH/BERT Speed Ratio']
    print("\n== COMPARATIVE METRICS ==")
    print(tabulate(comp, headers='keys', tablefmt='grid', showindex=False))

def plot_results(summary):
    datasets = summary['dataset'].values
    bar_width = 0.2
    x = np.arange(len(datasets))

    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.bar(x-bar_width, summary['blackhole_tokens'].astype(float), width=bar_width, label='Blackhole')
    plt.bar(x,       summary['gpt2_tokens'].astype(float), width=bar_width, label='GPT-2')
    plt.bar(x+bar_width, summary['bert_tokens'].astype(float), width=bar_width, label='BERT')
    plt.xticks(x, datasets, rotation=45)
    plt.title("Avg Token Counts"); plt.legend()

    plt.subplot(1,2,2)
    plt.bar(x-bar_width, summary['blackhole_time_ms'].astype(float), width=bar_width, label='Blackhole')
    plt.bar(x,          summary['gpt2_time_ms'].astype(float), width=bar_width, label='GPT-2')
    plt.bar(x+bar_width, summary['bert_time_ms'].astype(float), width=bar_width, label='BERT')
    plt.xticks(x, datasets, rotation=45)
    plt.title("Avg Tokenization Time (ms)"); plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    all_results = []
    for ds_config in DATASETS:
        samples = load_data_samples(ds_config, num_samples=25)
        results = run_benchmark(samples, ds_config["name"])
        all_results.extend(results)

    summary = generate_summary_table(all_results)
    print_summary_tables(summary)
    plot_results(summary)

if __name__ == "__main__":
    main()

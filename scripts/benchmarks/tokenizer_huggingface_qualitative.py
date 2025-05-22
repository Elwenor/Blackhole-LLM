from transformers import GPT2TokenizerFast, BertTokenizerFast
from datasets import load_dataset
import sys, os
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Union, Iterator

# Adjust sys.path to point to the directory containing blackhole.tokenizer_hugging_face
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    # Corrected import path for the new tokenizer
    from blackhole.tokenizer_hugging_face import BlackholeTokenizer
except ImportError as e:
    print(f"Error importing BlackholeTokenizer: {e}")
    print("Please ensure your BlackholeTokenizer is correctly located at 'blackhole\\tokenizer_hugging_face\\__init__.py' (if it's a package) or 'blackhole\\tokenizer_hugging_face\\your_module_name.py'.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


DATASETS = [
    {
        "name": "Scientific Text",
        "dataset": "ccdv/arxiv-summarization",
        "subset": None,
        "split": "train",
        "text_field": "article",
        "num_samples_for_train": 500 # Number of samples to use for tokenizer training from this dataset
    },
]

def print_summary(name, tokens, detok, original_text=None, original_metadata=None):
    print(f"\n=== {name} ===")
    if original_text is not None:
        print("Original text (first 500 chars):")
        print(original_text[:500] + ("..." if len(original_text) > 500 else ""))
        print("-" * 40)
    print("Detokenized text (first 500 chars):")
    print(detok[:500] + ("..." if len(detok) > 500 else ""))
    print(f"Total tokens: {len(tokens)}")
    # For GPT-2 and BERT, tokens are usually strings. For Blackhole, they are IDs.
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], int): # Blackhole's output (token IDs)
        print(f"Unique tokens (IDs): {len(set(tokens))}")
    else: # GPT-2/BERT's output (token strings)
        print(f"Unique tokens (Strings): {len(set(tokens))}")
    
    # Check exact match carefully due to potential whitespace or special token differences
    # Stripping space is a good common practice for comparison
    exact_match = detok.strip() == original_text.strip()
    print(f"Exact match to original (stripped): {exact_match}")

    if name.startswith("Blackhole") and original_metadata:
        # original_metadata is a list of (metadata_list, processed_to_original_map) tuples for batches
        # For a single sequence, we take the first element.
        if original_metadata and original_metadata[0] and original_metadata[0][0]:
            metadata_list_for_first_seq = original_metadata[0][0] # The list of metadata dicts
            
            # Count numbers detected by your _prepare_text_for_bpe_and_collect_metadata
            numbers_detected = sum(1 for m in metadata_list_for_first_seq if m.get('type') == 'NUM')
            print(f"Numbers detected by Blackhole's pre-processing: {numbers_detected}")
            

def get_data_for_benchmark(config, for_training=False):

    print(f"Loading data for dataset: {config['name']} (for {'training' if for_training else 'benchmarking'})")
    ds = load_dataset(config["dataset"], config.get("subset"), split=config["split"])

    if config.get("filter_func"):
        ds = ds.filter(config["filter_func"])
        if len(ds) == 0:
            raise ValueError(f"No samples found for filter in {config['name']} dataset after filtering")

    if for_training:
        # Return a generator for training texts
        # Ensure that `num_samples_for_train` is greater than 0 if training is intended
        num_samples = config["num_samples_for_train"]
        if num_samples == 0:
            print(f"Warning: num_samples_for_train is 0 for {config['name']}, no training data will be yielded.")
            return iter([]) # Return an empty iterator
            
        return (
            example[config["text_field"]] 
            for example in ds.shuffle(seed=42).filter(lambda x: x[config["text_field"]].strip() != '').select(range(min(len(ds), num_samples)))
        )
    else:
        # Return the first sample text for detailed display
        sample_text = ds[0][config["text_field"]]
        if len(sample_text) > 1500:
            sample_text = sample_text[:1500] + "... [truncated]\n"
        return sample_text


def main():
    # Load Hugging Face tokenizers once
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
    bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Instantiate BlackholeTokenizer once
    bh_tok = BlackholeTokenizer()
    print("\n--- Training BlackholeTokenizer ---")
    
    # Gather training texts from all datasets
    all_training_texts_iterator = get_data_for_benchmark(DATASETS[0], for_training=True)
    
    # Train the BlackholeTokenizer on the combined corpus
    train_start_time = time.time()
    bh_tok.train_tokenizer(all_training_texts_iterator, vocab_size=30000, min_freq=2, show_progress=True)
    train_end_time = time.time()
    print(f"--- BlackholeTokenizer trained in {train_end_time - train_start_time:.2f} seconds. ---")

    # It's good practice to save and reload after training, especially if you have custom logic
    # that might rely on the tokenizer being fully initialized from a saved state.
    # This mimics what your test_tokenizer.py does successfully.
    output_dir = "./blackhole_tokenizer_benchmark_trained"
    os.makedirs(output_dir, exist_ok=True)
    bh_tok.save_pretrained(output_dir)
    print(f"BlackholeTokenizer saved to {output_dir}")
    
    # Reload the tokenizer to ensure consistency, as in your test file
    loaded_bh_tok = BlackholeTokenizer.from_pretrained(output_dir)
    print(f"BlackholeTokenizer loaded from {output_dir}")


    for ds_conf in DATASETS:
        # Get one sample text for detailed qualitative analysis for the current dataset
        sample_text = get_data_for_benchmark(ds_conf, for_training=False)
        
        if not sample_text.strip():
            print(f"Skipping empty sample text for {ds_conf['name']}")
            continue

        # --- Blackhole Tokenize/Detokenize ---
        print(f"\n--- Processing Blackhole Tokenizer for {ds_conf['name']} ---")
        # Use the loaded tokenizer, which is confirmed to work in your test script
        bh_enc = loaded_bh_tok(sample_text, add_special_tokens=True, truncation=True, max_length=512, return_tensors=None)
        bh_tokens_ids = bh_enc["input_ids"] 

        bh_detok_with_special = loaded_bh_tok.decode(bh_tokens_ids, skip_special_tokens=False)
        bh_detok_clean = loaded_bh_tok.decode(bh_tokens_ids, skip_special_tokens=True)
        
        # Pass the loaded_bh_tok to print_summary so it can access its internal states
        print_summary(
            f"Blackhole ({ds_conf['name']}) - With Special Tokens", 
            bh_tokens_ids, 
            bh_detok_with_special, 
            original_text=sample_text, 
            original_metadata=loaded_bh_tok._last_original_metadata_for_decode # Use loaded_bh_tok's metadata
        )
        print_summary(
            f"Blackhole ({ds_conf['name']}) - Clean", 
            bh_tokens_ids, 
            bh_detok_clean, 
            original_text=sample_text, 
            original_metadata=loaded_bh_tok._last_original_metadata_for_decode # Use loaded_bh_tok's metadata
        )


        # --- GPT-2 Tokenize/Detokenize ---
        print(f"\n--- Processing GPT-2 Tokenizer for {ds_conf['name']} ---")
        gpt2_tokens_ids = gpt2_tok.encode(sample_text)
        gpt2_detok = gpt2_tok.decode(gpt2_tokens_ids)
        gpt2_tokens_str = gpt2_tok.convert_ids_to_tokens(gpt2_tokens_ids)
        print_summary(f"GPT-2 ({ds_conf['name']})", gpt2_tokens_str, gpt2_detok, original_text=sample_text)

        # --- BERT Tokenize/Detokenize ---
        print(f"\n--- Processing BERT Tokenizer for {ds_conf['name']} ---")
        bert_enc = bert_tok.encode_plus(sample_text, add_special_tokens=False, truncation=True, max_length=512)
        bert_tokens_ids = bert_enc["input_ids"]
        bert_detok = bert_tok.decode(bert_tokens_ids)
        bert_tokens_str = bert_tok.convert_ids_to_tokens(bert_tokens_ids)
        print_summary(f"BERT ({ds_conf['name']})", bert_tokens_str, bert_detok, original_text=sample_text)


if __name__ == "__main__":
    main()
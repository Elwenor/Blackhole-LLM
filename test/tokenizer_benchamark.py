from transformers import GPT2TokenizerFast, BertTokenizerFast
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blackhole.tokenizer import tokenize, detokenize

text = "On 2023-07-15, the stock price jumped from $1,234.56 to $1,567.89, while 0x2F4A was logged at 14:30."

def print_summary(name, tokens, detok, show_map=False):
    print(f"\n{name}: ")
    print("Detokenized:", detok)
    print("Total tokens:", len(tokens))
    print("Unique tokens:", len(set(tokens)))
    if show_map:
        print("\nNumber Map (token index → (value, type, raw)):")
        for idx, (val, typ, raw) in number_map.items():
            print(f"{idx}: {val} ({typ}), raw: {raw}")

# --- Blackhole ---
bh_tokens, number_map = tokenize(text)
bh_detok = detokenize(bh_tokens, number_map)
print_summary("Blackhole", bh_tokens, bh_detok, show_map=True)

# --- GPT-2 ---
gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
gpt2_tokens = gpt2_tok.tokenize(text)
gpt2_detok = gpt2_tok.convert_tokens_to_string(gpt2_tokens)
print_summary("GPT-2", gpt2_tokens, gpt2_detok)

# --- BERT ---
bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_tokens = bert_tok.tokenize(text)
bert_detok = bert_tok.convert_tokens_to_string(bert_tokens)
print_summary("BERT", bert_tokens, bert_detok)

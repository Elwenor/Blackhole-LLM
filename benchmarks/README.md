## Detokenization Comparison Example
> Note:
>The detokenization feature is currently under active development. While it already reconstructs the original text >with good accuracy—including numeric formatting and spacing—it may still produce minor formatting inconsistencies. >Improvements are ongoing to ensure even better output fidelity in future releases.

```python
from transformers import GPT2TokenizerFast, BertTokenizerFast
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

# Blackhole
bh_tokens, number_map = tokenize(text)
bh_detok = detokenize(bh_tokens, number_map)
print_summary("Blackhole", bh_tokens, bh_detok, show_map=True)

# GPT-2
gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
gpt2_tokens = gpt2_tok.tokenize(text)
gpt2_detok = gpt2_tok.convert_tokens_to_string(gpt2_tokens)
print_summary("GPT-2", gpt2_tokens, gpt2_detok)

# BERT
bert_tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_tokens = bert_tok.tokenize(text)
bert_detok = bert_tok.convert_tokens_to_string(bert_tokens)
print_summary("BERT", bert_tokens, bert_detok)
```
### Output Benchmark:

```
## Blackhole
- **Detokenized:**  
  On 2023-07-15, the stock price jumped from $1,234.56 to $1,567.89, while 0x2F4A was logged at 14:30.
- **Total tokens:** 36  
- **Unique tokens:** 19  

### Number Map (token index → (value, type, raw)):
- 2: 2023 (int), raw:  2023  
- 4: 7 (int), raw:  07  
- 6: 15 (int), raw:  15  
- 19: 1234.56 (float), raw: 1,234.56  
- 23: 1567.89 (float), raw: 1,567.89  
- 26: 12106 (hex), raw:  0x2F4A  
- 32: 14 (int), raw:  14  
- 34: 30 (int), raw:  30

---
## GPT-2
- **Detokenized:**  
  On 2023-07-15, the stock price jumped from $1,234.56 to $1,567.89, while 0x2F4A was logged at 14:30.
- **Total tokens:** 42  
- **Unique tokens:** 34  

---

## BERT
- **Detokenized:**  
  on 2023 - 07 - 15, the stock price jumped from $ 1, 234. 56 to $ 1, 567. 89, while 0x2f4a was logged at 14 : 30.
- **Total tokens:** 42  
- **Unique tokens:** 33
```

# Comparison of Blackhole Tokenizer with GPT-2 and BERT

This example demonstrates how different tokenizers handle complex numeric formats such as dates, floats with commas, hexadecimal numbers, and timestamps.

## Blackhole Tokenizer

Blackhole replaces numeric values with a special abstraction, effectively substituting all numbers with a unified token (represented here conceptually) and maintaining a separate number map. This approach:

- Reduces the total number of unique tokens significantly (19 unique tokens vs. 33-34 in other tokenizers).
- Preserves numeric values as single conceptual units rather than fragmenting them into multiple tokens (e.g., `1,234.56` is treated as one float value, not split into `1`, `,`, `234`, `.`, `56`).
- Handles hexadecimal values and timestamps similarly, keeping them intact in the number map.

>However, this abstraction is not perfect. For example, dates such as `2023-07-15` are tokenized into separate integer tokens `2023`, `7`, and `15`, losing the direct connection to the original date format. So while numeric values are extracted and abstracted, >the tokenization does not preserve the full original formatting in all cases.

## GPT-2 and BERT Tokenizers

Both GPT-2 and BERT tokenize numeric data by splitting on formatting characters. This results in:

- Higher total token counts (42 tokens in the example).
- A larger number of unique tokens (33–34 unique tokens).
- Fragmentation of numeric values into multiple tokens (e.g., splitting numbers with commas, dots, or hex prefixes).
- Less efficient vocabulary usage, as each numeric component is stored as a separate token.



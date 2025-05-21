# Blackhole-LLM

Blackhole-LLM — An experimental Python framework for building and customizing large language models in Python, leveraging `torch` (with a roadmap toward full PyTorch integration). It features a tailored `GPT2TokenizerFast`, custom dual embeddings (textual + numerical), and a strong focus on enhancing mathematical reasoning and structured input handling.

> This project is under active development.  
> It's public for transparency and feedback, but not yet intended for production use.

---

## Features

- Custom tokenizer based on `GPT2TokenizerFast`, adapted for better handling of numeric, symbolic, and mathematical input  
- Improved tokenization efficiency: reduces the number of unique tokens compared to standard tokenizers (e.g. GPT-2, BERT) while maintaining logical structure  
- Dual embedding architecture supporting both textual and numerical feature encoding  
- Framework-level integration with `torch` (early-stage)  
- Modular layout with internal benchmarks and unit tests for tokenizer and embedding behavior

---

# Tokenizer

The tokenizer is a custom extension of `GPT2TokenizerFast`, designed specifically for efficient processing of texts containing numeric structures, mathematical symbols, and formatted input.

---

## How it works

- Introduces special tokens to mark key elements:

  - `<|num|>` — marks all numbers (integers, floats, hexadecimals, dates, times), enabling uniform numeric data handling.  
  - `<|cap|>` — marks capitalization at the start of words, avoiding vocabulary bloat from case variants.  
  - `<|space|>` — explicit tokens representing spaces, helping preserve original formatting.  
  - Mathematical symbols like π, ∞, √, ± are also treated as special tokens.

- Uses **regular expressions** to detect complex numeric patterns such as:

  - Hexadecimal numbers (`0x...`)  
  - Dates (`YYYY-MM-DD`, `YYYY/MM/DD`)  
  - Times (`HH:MM`, `HH:MM:SS`)  
  - Standard integers, floats, and scientific notation (e.g., `1.2e-4`).

- Upon matching these patterns, replaces them with the `<|num|>` token while storing the original values and types separately, preserving precision without inflating token count.

- Words starting with a capital letter (pattern: uppercase letter + lowercase letters) are prefixed with `<|cap|>`, then lowercased to keep capitalization information separate.

- Spaces between tokens are explicitly encoded as `<|space|>`, improving the model’s understanding of text structure.

- Additionally, the tokenizer cleans and normalizes input by removing extraneous whitespace and fixing common number formatting issues (e.g., misplaced commas or dots).

---

## Benefits

- **Reduced vocabulary size** compared to standard tokenizers (GPT-2, BERT) thanks to unified numeric tokenization and capitalization tokens.  
- **Improved representation of mathematical and structured text**, crucial for tasks involving numbers, formulas, or dates.  
- Enables **accurate reconstruction** of the original text, preserving formatting, capitalization, and numeric precision.

---

## Limitations

- Tailored for highly structured and numeric-heavy text; less optimized for casual conversational language.  
- Complex or ambiguous numeric formats may require further refinement.

---

## Tokens output example

```python
from blackhole.tokenizer import tokenize

text = """
The price rose from $1,234.56 on 2023-05-20 to 0x1A3F units by 12:30 PM. Meanwhile, the experimental drug reduced the virus count by 0.000123 units ...
"""

tokens, number_map = tokenize(text)
print("Tokens:", tokens)
```
Special tokens like <|num|>, <|cap|>, and <|space|> are used to encode numbers, capitalization, and spaces explicitly. This structured representation helps the model better understand numeric and formatted data.

### Output:

```plaintext
Tokens: ['<|cap|>', 'the', '<|space|>', 'price', '<|space|>', 'rose', '<|space|>', 'from', '<|space|>', '$', '<|num|>', 'on', '<|num|>', '-', '<|num|>', '-', '<|num|>'...
```


## Tokens numbermap example

```python
from blackhole.tokenizer import tokenize

text = """
The price rose from $1,234.56 on 2023-05-20 to 0x1A3F units by 12:30 PM. Meanwhile, the experimental drug reduced the virus count by 0.000123 units ...
"""

print("\nNumber Map (token index → (value, type, raw)):")
for idx, (val, typ, raw) in number_map.items():
    print(f"{idx}: {val} ({typ}), raw: {raw}")

```

### Output:

```plaintext
Number Map (token index → (value, type, raw)):
10: 1234.56 (float), raw: 1,234.56
12: 2023 (int), raw:  2023
14: 5 (int), raw:  05
16: 20 (int), raw:  20
18: 6719 (hex), raw:  0x1A3F
22: 12 (int), raw:  12
24: 30 (int), raw:  30
45: 0.000123 (float), raw:  0.000123
```
This output shows how each `<|num|>` token in the tokenized sequence maps to its original numeric value and format.

- **Token index**: position of the `<|num|>` token in the token list.  
- **Value and type**: exact numeric value parsed from the text, classified as float, int, or hex.  
- **Raw**: original number string as it appeared, preserving formatting like commas, leading zeros, or hex notation.

For example, `1,234.56` is tokenized as `<|num|>` at index 10, stored as the float `1234.56` with original formatting preserved. Dates and times are tokenized as multiple numeric tokens corresponding to their components (year, month, day, hour, minute).

This design:
- Keeps the tokenizer vocabulary compact by using a single `<|num|>` token for all numbers.  
- Preserves numeric precision and formatting for accurate reconstruction (detokenization).  
- Enables the model to better understand and reason about numeric and structured data.

## Tokens token sumary and unique token example

```python
from blackhole.tokenizer import tokenize, summarize_tokens

text = """
The price rose from $1,234.56 on 2023-05-20 to 0x1A3F units by 12:30 PM. Meanwhile, the experimental drug reduced the virus count by 0.000123 units ...
"""

tokens, number_map = tokenize(text)

print("\nToken Summary:")
for tok, idx, count in summarize_tokens(tokens):
    print(f"ID: {idx:2d} | Token: '{tok}' | Count: {count}")

unique_tokens = len(set(tokens))
print(f"\nNumber of unique tokens: {unique_tokens}")
```
### Output:
```
Token Summary:
ID:  0 | Token: '<|cap|>' | Count: 2
ID:  1 | Token: 'the' | Count: 3
ID:  2 | Token: '<|space|>' | Count: 12
ID:  3 | Token: 'price' | Count: 1
ID:  4 | Token: 'rose' | Count: 1
ID:  5 | Token: 'from' | Count: 1
ID:  6 | Token: '$' | Count: 1
...

Number of unique tokens: 23
```

This example shows how the tokenizer handles complex numeric formats by replacing all numbers with a unique `<|num|>` token. This means:

- Numeric formatting like commas, leading zeros, hex notation, or scientific notation is **not broken into multiple tokens**, preserving the original number as a single conceptual unit.  
- This drastically **reduces the number of unique tokens** needed in the vocabulary, keeping it compact and efficient.  
- As seen in the output, the `<|num|>` token frequently appears, representing all numeric values uniformly.

### Trade-offs

**Pros:**  
- Smaller vocabulary  
- Better numeric precision  
- Easier reasoning about numbers in the model  

**Cons:**  
- The tokenizer abstracts away the numeric value into a single token, so downstream tasks must rely on the number map for precise numeric data; this can complicate some use cases if not handled properly.

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
### Output:

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

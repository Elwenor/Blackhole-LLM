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

### Tokens numbermap example

```python
from blackhole.tokenizer import tokenize

...

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

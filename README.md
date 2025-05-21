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

## Tokenizer

The tokenizer is a custom extension of `GPT2TokenizerFast` tailored to efficiently handle structured, numerical, and mathematical text.

### How it works

- **Special tokens** are introduced for key elements:  
  - `<|num|>` marks all numbers (integers, floats, hexadecimals, dates, times), enabling uniform treatment of numeric data.  
  - `<|cap|>` indicates capitalization, avoiding bloated vocabularies due to case variants.  
  - `<|space|>` preserves spaces explicitly to maintain input formatting.  
  - Mathematical symbols like π, ∞, √, ± are also treated as special tokens.

- The tokenizer uses **regular expressions** to detect complex numeric patterns such as:  
  - Hexadecimal numbers (`0x...`)  
  - Dates (`YYYY-MM-DD` or `YYYY/MM/DD`)  
  - Times (`HH:MM` or `HH:MM:SS`)  
  - Standard integers and floating-point numbers, including scientific notation.

- Upon finding these patterns, it replaces them with `<|num|>` tokens, while storing the original numeric values and types separately. This allows the model to maintain precise numeric information without inflating token counts.

- Words starting with uppercase followed by lowercase letters trigger the insertion of `<|cap|>` tokens before their lowercase form, capturing capitalization without duplicating tokens.

- Spaces between tokens are encoded explicitly as `<|space|>` tokens, improving the model's understanding of text structure and spacing.

- The tokenizer also cleans and normalizes input by removing extraneous whitespace and fixing common number formatting issues (like misplaced commas or dots).

### Benefits

- **Reduced vocabulary size** compared to standard tokenizers like GPT-2 or BERT, thanks to unified numeric tokenization and capitalization tokens.  
- **Improved mathematical and structured text representation**, essential for tasks involving numbers, formulas, or dates.  
- Facilitates **accurate reconstruction** of original text with detokenization, preserving formatting, capitalization, and numeric precision.

### Limitations

- Tailored for structured and numeric-heavy text; less optimized for casual conversational language.  
- Complex or ambiguous numeric formats might require further refinement.

---

## Usage Example

Below is a complete, step-by-step example of using the tokenizer on a sample text, demonstrating tokenization, number mapping, token summary, and detokenization in one place.

### Sample Input Text

```python
text = """
The experimental drug reduced the virus count by 0.000123 units, which is a 99.999% improvement compared to the previous 1.2e-4 baseline. The lab reported 12,345 samples processed, with an error margin of ±0.0001.
"""

What happens here:

The text is scanned for words, punctuation, and complex numbers (floats, scientific notation, hexadecimals, dates, times).

Numbers get replaced by a special <|num|> token to standardize representation.

Words starting with a capital letter are flagged with <|cap|>, then lowercased, preserving capitalization info separately.

Spaces between tokens are explicitly encoded as <|space|> tokens to avoid losing spacing info.

The result is a logically structured token list ready for model consumption.

Sample token output (truncated):

plaintext
Kopiuj
Edytuj
['<|cap|>', 'the', '<|space|>', 'experimental', '<|space|>', 'drug', '<|space|>', 'reduced', '<|space|>', 'the', '<|space|>', 'virus', ..., '<|num|>', 'units', ',', 'which', '<|space|>', 'is', ...]
2. Number Map
python
Kopiuj
Edytuj
print("\nNumber Map (token index → (value, type, raw)):")
for idx, (val, typ, raw) in number_map.items():
    print(f"{idx}: {val} ({typ}), raw: {raw}")
Explanation:

Each <|num|> token has an associated entry in number_map.

The map stores a tuple: (value, type, raw_text)

value — numeric value parsed (float, int, or hex).

type — number category.

raw_text — original number string as it appeared (preserving formatting like commas or scientific notation).

This allows precise numerical info to be preserved and reused downstream.

Example output:

plaintext
Kopiuj
Edytuj
16: 0.000123 (float), raw:  0.000123
24: 99.999 (float), raw:  99.999
36: 0.00012 (float), raw:  1.2e-4
45: 12345 (int), raw:  12,345
61: 0.0001 (float), raw: 0.0001
3. Token Summary
python
Kopiuj
Edytuj
for tok, idx, count in tokenizer.summarize_tokens(tokens):
    print(f"ID: {idx:2d} | Token: '{tok}' | Count: {count}")
What it does:

Counts all unique tokens and their frequencies.

Assigns consistent IDs to tokens in order of appearance.

Useful for analyzing token distribution and frequency, essential in vocabulary design or debugging.

Sample output snippet:

plaintext
Kopiuj
Edytuj
ID:  0 | Token: '<|cap|>' | Count: 2
ID:  1 | Token: 'the' | Count: 4
ID:  2 | Token: '<|space|>' | Count: 22
ID:  9 | Token: '<|num|>' | Count: 5
ID: 31 | Token: '±' | Count: 1
...
4. Detokenization
python
Kopiuj
Edytuj
print(tokenizer.detokenize(tokens, number_map))
What happens:

Reverses tokenization, reconstructing readable text.

Inserts original numeric values from number_map instead of <|num|> tokens.

Correctly restores capitalization, spaces, and punctuation.

Output is very close

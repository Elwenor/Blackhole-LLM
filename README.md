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

## Usage Example

```python
text = """
The experimental drug reduced the virus count by 0.000123 units, which is a 99.999% improvement compared to the previous 1.2e-4 baseline. The lab reported 12,345 samples processed, with an error margin of ±0.0001.
"""
```
What happens here:

- The text is scanned for words, punctuation, and complex numbers (floats, scientific notation, hexadecimals, dates, times).

- Numbers get replaced by a special `<|num|>` token to standardize representation.

- Words starting with a capital letter are flagged with `<|cap|>`, then lowercased, preserving capitalization info separately.

- Spaces between tokens are explicitly encoded as `<|space|>` tokens to avoid losing spacing info.

- The result is a logically structured token list ready for model consumption.


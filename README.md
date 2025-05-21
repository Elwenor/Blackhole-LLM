# Blackhole-LLM

Blackhole-LLM — An experimental Python framework for building and customizing large language models in Python, leveraging `torch` (with a roadmap toward full PyTorch integration). It features a tailored `GPT2TokenizerFast`, custom dual embeddings (textual + numerical), and a strong focus on enhancing mathematical reasoning and structured input handling.

>This project is under active development.  
>It's public for transparency and feedback, but not yet intended for production use.

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

### Basic Usage Example

```python
from blackhole.tokenizer import BlackholeTokenizer

tokenizer = BlackholeTokenizer()
tokens, number_map = tokenizer.tokenize("The mass of the black hole is 10^30 kg.")
print(tokens)



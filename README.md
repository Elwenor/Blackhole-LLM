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

The tokenizer is a customized extension of `GPT2TokenizerFast`, specifically designed to handle structured and numerical text inputs with higher efficiency and precision.

It introduces several innovations:

- **Special tokens** to explicitly mark numbers (`<|num|>`), capitalization (`<|cap|>`), spaces (`<|space|>`), and common mathematical symbols (`∞`, `π`, `√`, `≈`, `±`).
- **Pattern matching** to detect and isolate numerical data types such as integers, floats, hexadecimals, dates, and times, converting them into uniform tokens while preserving their original values and types separately.
- **Capitalization handling** via dedicated tokens that track case changes without inflating vocabulary size.
- **Explicit spacing tokens** that preserve input structure critical for scientific and mathematical texts.
- During detokenization, stored metadata ensures the exact reconstruction of the original text, including spacing, capitalization, and numeric precision.

### Why this matters

Standard tokenizers like GPT2 or BERT tend to fragment numeric and mathematical content into many unique tokens, inflating vocabulary size and complicating model learning. This tokenizer reduces fragmentation, making numeric reasoning and structured input processing more efficient and reliable.

### Limitations

- Best suited for structured and mathematical text rather than conversational language.
- May require tuning to cover edge cases in complex numeric/date formats.

### Basic Usage

```python
from blackhole.tokenizer import BlackholeTokenizer

tokenizer = BlackholeTokenizer()
tokens = tokenizer.encode("The mass of the black hole is 10^30 kg.")
print(tokens)


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

The tokenizer is a customized version of `GPT2TokenizerFast`, adapted for structured and numerical data. It aims to optimize token distribution and reduce unnecessary fragmentation — particularly important for mathematical inputs.

### Basic Usage

```python
from blackhole.tokenizer import BlackholeTokenizer

tokenizer = BlackholeTokenizer()
tokens = tokenizer.encode("The mass of the black hole is 10^30 kg.")
print(tokens)

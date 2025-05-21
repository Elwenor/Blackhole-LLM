# Blackhole-LLM

Blackhole-LLM — An experimental Python framework for building and customizing large language models, featuring a tailored `GPT2TokenizerFast`, custom dual embeddings (textual + numerical), and a strong focus on enhancing mathematical reasoning.

>This project is under active development.  
>It's public for transparency and feedback, but not yet intended for production use.

---

## Features

- Custom `GPT2TokenizerFast` implementation for domain-specific preprocessing
- Dual embedding architecture (text + numerical features)
- Focus on improving mathematical and structured reasoning in LLMs
- Lightweight benchmarking and testing scripts included

---

## Usage Example

### Tokenizer

```python
from blackhole.tokenizer import BlackholeTokenizer

tokenizer = BlackholeTokenizer()
tokens = tokenizer.encode("The mass of the black hole is 10^30 kg.")
print(tokens)

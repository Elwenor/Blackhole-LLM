# 🌌 Blackhole-LLM

**Blackhole-LLM** — An experimental Python framework for building and customizing large language models, featuring a tailored `GPT2TokenizerFast`, custom dual embeddings (textual + numerical), and a strong focus on enhancing mathematical reasoning.

> ⚠️ This project is under heavy development.  
> It's public for visibility and discussion, but **not production-ready** or intended for external use (yet).

---

## 📦 Features

- ✅ Custom `GPT2TokenizerFast` tailored for numeric-heavy data
- ✅ Dual embedding architecture — combines **text** and **number** embeddings
- ✅ Designed for **LLM mathematical reasoning**
- 🧪 Built-in test scripts for benchmarking components

---

## 🚀 Usage Example

Below is an example of how to use the tokenizer and embeddings inside your Python environment:

### 🔹 Tokenizer Example

```python
from blackhole.tokenizer import BlackholeTokenizer

tokenizer = BlackholeTokenizer()
tokens = tokenizer.encode("The mass of the black hole is 10^30 kg.")
print(tokens)

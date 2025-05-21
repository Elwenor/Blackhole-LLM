# Blackhole-LLM

Blackhole-LLM — An experimental Python framework for building and customizing large language models in Python, leveraging `torch` (with a roadmap toward full PyTorch integration). It features a tailored `GPT2TokenizerFast`, custom dual embeddings (textual + numerical), and a strong focus on enhancing mathematical reasoning and structured input handling.

> This project is under active development.  
> It's public for transparency and feedback, but not yet intended for production use.

> ! IT CURRENTLY HAS A WORKING TOKENIZER AND ITS EMBEDDING. CREATING TEST MODELS IS IN THE DEVELOPMENT STAGE !

> ! LOCAL BENCHMARK NLP AVAILABLE. GPT2 TOKENIZER + BERT vs BLACKHOLE !

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

  - `<|num|>` — marks all numbers (integers, floats, hexadecimals, dates, times), enabling uniform numeric data handling. This acts as a **placeholder** token replacing the actual numeric literal while its exact value is stored separately to maintain precision and avoid vocabulary explosion.
  
  - `<|cap|>` — marks capitalization at the start of words, avoiding vocabulary bloat from case variants by lowering the base token but preserving case information via this prefix token.
  
  - `<|space|>` — explicit tokens representing spaces, helping preserve original formatting and sentence structure, which is often lost in classical tokenization.
  
  - Mathematical symbols like π, ∞, √, ± are also treated as special tokens to capture their semantic role cleanly.

- Uses **regular expressions** to detect complex numeric patterns such as:

  - Hexadecimal numbers (`0x...`)  
  - Dates in formats like `YYYY-MM-DD` or `YYYY/MM/DD`  
  - Times like `HH:MM` or `HH:MM:SS`  
  - Standard integers, floats, and scientific notation (e.g., `1.2e-4`).

- Upon matching these patterns, replaces them with the `<|num|>` token while storing the original numeric value and its type separately. This lets the tokenizer maintain **precision and semantics** without inflating the token count.

- Words starting with a capital letter (pattern: uppercase letter + lowercase letters) are prefixed with `<|cap|>`, then lowercased, which prevents vocabulary size inflation from multiple capitalized variants.

- Spaces between tokens are explicitly encoded as `<|space|>`, improving the model’s understanding of text formatting and sentence structure, often crucial for structured input like formulas or tabular data.

- The tokenizer also performs normalization, removing extraneous whitespace and fixing common number formatting issues (e.g., misplaced commas or dots), ensuring consistent tokenization.

---

## Benefits

- **Reduced vocabulary size:**  
  By unifying all numeric forms under a single `<|num|>` token and separating case information via `<|cap|>`, the tokenizer avoids bloating the vocabulary with variants of numbers or capitalized words.

- **Enhanced numeric and symbolic handling:**  
  Specialized handling of numbers, dates, times, and math symbols enables better downstream performance on tasks involving mathematical reasoning, scientific text, or data with embedded numerics.

- **Explicit preservation of formatting:**  
  Introducing `<|space|>` tokens maintains the original input’s spacing, which helps models learn formatting cues and improves reconstruction quality.

- **Improved detokenization fidelity:**  
  Storing original numeric values allows exact reversal of tokenization, preserving precision and making the model’s output easier to interpret and trust.

- **Modular and extendable:**  
  The tokenizer’s design separates numeric processing from text tokenization, allowing for easier updates or adaptation to new numeric formats or domains.

---

## Limitations and Challenges

- **Complexity and performance:**  
  The heavy use of regex matching and multi-step token insertion can slow tokenization compared to highly optimized byte-level tokenizers like standard GPT2. This can be a bottleneck for large-scale data preprocessing.

- **Ambiguous or edge-case numeric formats:**  
  Real-world data often includes ambiguous notations, OCR errors, or locale-dependent formats (e.g., decimal commas, varied date formats) that require continuous regex refinement or more advanced parsing.

- **Not optimized for casual or conversational text:**  
  The tokenizer’s strong focus on numeric and structured input means it might be less efficient or overly complex for pure natural language data without many numbers or symbols.

- **Handling of capitalization is simplistic:**  
  Only initial capitalization is tagged via `<|cap|>`, so words in all caps (acronyms) or mixed case are not distinctly handled, potentially losing some nuance.

- **Serialization and batching challenges:**  
  The separate `number_map` structure that stores original numeric values needs careful handling in batch processing and training pipelines to maintain alignment between tokens and numeric embeddings.

---

## Potential Improvements

- **Stepwise tokenization pipeline:**  
  Splitting regex processing into stages (dates → times → hex → numbers) might improve clarity, maintainability, and speed.

- **Selective `<|space|>` insertion:**  
  Only add `<|space|>` where critical (e.g., between words) to reduce token overhead and simplify sequences.

- **Extended capitalization tagging:**  
  Introduce tags for ALL CAPS or mixed case to better capture acronyms and proper nouns.

- **Advanced numeric parsing:**  
  Integrate domain-specific numeric parsers or libraries (e.g., `dateutil`, `regex` Unicode features) to capture edge cases and locale variants.

- **Embedding API unification:**  
  Build an interface that cleanly merges textual embeddings from GPT2 tokens with numeric embeddings derived from stored numeric values, using cross-attention and alignment losses to fuse representations.

- **Performance optimizations:**  
  Consider reimplementing tokenization logic in faster languages (Rust/Cython) or leveraging HuggingFace’s `tokenizers` library for speed gains.

- **Robust serialization:**  
  Ensure the `number_map` and token sequences can be efficiently serialized and fed into training pipelines, especially when batching multiple sequences.

---

## Summary

Blackhole-LLM’s tokenizer represents an **innovative and principled approach** to bridging standard NLP tokenization with the specialized needs of numeric and mathematical text. Its core idea—separating numeric content from textual tokens via placeholder tokens and explicit capitalization and spacing marks—is a clever way to reduce vocabulary size while preserving crucial semantic details.

However, this approach also **comes with trade-offs**: additional complexity, performance overhead, and the need for continuous regex tuning and data-specific adjustments. The tokenizer is most promising in domains requiring **precise mathematical or structured input understanding** but is less suited for generic natural language tasks without numeric density.

In short: Blackhole-LLM’s tokenizer sets a solid foundation for **numerically-aware LLM tokenization**, but to unlock its full potential, it requires ongoing refinement in speed, robustness, and integration with dual embedding architectures that jointly model text and numbers.

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
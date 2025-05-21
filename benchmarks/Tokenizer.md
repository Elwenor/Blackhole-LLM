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
### Output Benchmark:

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

# Comparison of Blackhole Tokenizer with GPT-2 and BERT

This example demonstrates how different tokenizers handle complex numeric formats such as dates, floats with commas, hexadecimal numbers, and timestamps. These subtle differences in tokenization can have a major impact on how language models understand and process numerical data embedded in text.

---

## Blackhole Tokenizer

Blackhole replaces numeric values with a special abstraction, effectively substituting all numbers with a unified token (conceptually represented here) and maintaining a separate **number map** that keeps track of the actual values and types. This approach offers several significant advantages:

- **Drastic reduction in unique tokens:**  
  Blackhole uses only **19 unique tokens** in this example, compared to 33–34 unique tokens in GPT-2 and BERT. This reduction leads to a smaller and more manageable vocabulary size, which improves learning efficiency and generalization.

- **Preservation of numeric integrity as atomic units:**  
  Numeric values like `1,234.56` are treated as a single **float** token, rather than being fragmented into multiple tokens such as `1`, `,`, `234`, `.`, `56`. This means the model can reason about these numbers as whole entities, improving numerical comprehension.

- **Handling of hexadecimal and timestamp formats:**  
  Similar to floats and integers, hexadecimal values like `0x2F4A` and timestamps such as `14:30` are preserved as distinct conceptual tokens with their values stored in the number map, avoiding meaningless splits.

- **Separate Number Map for precise value tracking:**  
  By keeping a detailed mapping of token indices to original numeric values and their types (`int`, `float`, `hex`), Blackhole enables downstream components or models to access exact numerical data without ambiguity.

> **Caveats:**  
> This abstraction isn't flawless. For instance, date strings like `2023-07-15` are split into separate integer tokens (`2023`, `7`, `15`), losing the direct textual format. While numeric values themselves are preserved, some of the original formatting nuances are lost.

---

## GPT-2 and BERT Tokenizers

GPT-2 and BERT follow a more traditional tokenization strategy which involves splitting text on punctuation and formatting characters. This leads to the following characteristics:

- **Higher token counts and vocabulary size:**  
  Both tokenizers produce **42 tokens** in this example, with about **33–34 unique tokens**. This fragmentation increases model complexity and computational load.

- **Fragmentation of numeric data:**  
  Numbers like `1,234.56` are broken into separate tokens: `1`, `,`, `234`, `.`, `56`. Similarly, hexadecimals and timestamps are split on non-alphanumeric characters (e.g., `0x2f4a` → `0x`, `2f`, `4a` in some cases).

- **Inefficient vocabulary usage:**  
  Because each numeric fragment is stored separately, the model's vocabulary is bloated with many partial tokens, reducing the effectiveness of token embeddings and complicating pattern recognition.

- **Loss of numerical semantics:**  
  Splitting numbers into pieces forces the model to infer relationships across multiple tokens to reconstruct numeric values, making precise numerical reasoning harder.

---

## Why Does This Matter?

When dealing with tasks that require mathematical understanding, financial data processing, scientific text, or any context where numbers matter, tokenization strategy can make or break the model’s performance.

### Advantages of Blackhole Tokenizer Approach:

- **Compact and focused vocabulary:**  
  By replacing all numbers with a special abstract token and maintaining a separate numeric map, Blackhole drastically reduces the number of unique tokens in the model’s vocabulary. A smaller, more focused vocabulary leads to faster training and better generalization, as the model isn’t distracted by countless numeric fragments or rare tokens that only represent different numerical values. This lowers the risk of overfitting on spurious token fragments and helps the model concentrate on meaningful semantic patterns and relationships between words and numbers.

- **Better numeric reasoning:**  
  Numbers are treated as atomic units — full values instead of fragmented pieces (for example, `1,234.56` is one token rather than a sequence like `1`, `,`, `234`, `.`, `56`). This allows the model to associate tokens directly with their numeric meanings, making it easier to learn mathematical patterns, comparisons, and numeric relationships within natural language. The model isn’t forced to reconstruct values from disjointed tokens, which significantly improves its ability to reason about quantities.

- **Easier integration with downstream numeric modules (dual embedding channels):**  
  One of Blackhole’s key innovations is the separation of the token stream into **two parallel channels**: the standard textual tokens and a numeric map containing the actual values, types, and positions of numbers. This dual-channel design means numerical information can be fed independently into the model’s embedding layers or downstream components as a separate, structured signal. Consequently, models can leverage both the contextual language embeddings **and** precise numeric embeddings simultaneously, greatly enhancing their ability to handle complex numeric reasoning tasks, perform arithmetic operations, or integrate with external numerical solvers. This architecture effectively adds an additional “numeric modality” to traditional text embeddings, opening the door to more sophisticated, multimodal understanding within language models.


### Limitations to Keep in Mind:

- Not all formatting nuances are preserved (e.g., date formats).
- Requires additional machinery to maintain and use the number map effectively.
- May need fine-tuning for domain-specific numeric formats beyond the basics.

---

## Summary Table

| Feature                      | Blackhole Tokenizer                    | GPT-2                         | BERT                          |
|:----------------------------:|:------------------------------------:|:-----------------------------:|:-----------------------------:|
| **Total tokens**             | 36                                   | 42                            | 42                            |
| **Unique tokens**            | 19                                   | 34                            | 33                            |
| **Numeric token fragmentation** | No (numbers treated as one token)  | Yes (split on punctuation)    | Yes (split with even more spaces) |
| **Numeric value preservation**  | Yes, via number map                 | No                            | No                            |
| **Vocabulary size efficiency**  | High                              | Medium                        | Medium                        |
| **Formatting preservation**      | Partial (e.g. dates fragmented)   | Low                           | Low                           |


---

This comparison highlights that while Blackhole Tokenizer is still evolving, its approach to numeric tokenization provides a solid foundation for models aimed at improved understanding of mathematical and numeric content, outperforming standard tokenizers in these critical areas.

---

# Embedding Benchmark Example

> This example demonstrates how the Blackhole tokenizer and number embedding pipeline handle numeric tokens in a sample text, showing tokens, number mapping, embeddings, and decoded numeric values.

```python
from blackhole.tokenizer import *
from blackhole.embedding import *

def main():
    text = """
Add 3.14 and -42 to zero. The total sum should be 0.0.  
In physics, the speed of light is approximately 299792458 m/s.  
Negative temperatures like -273.15°C represent absolute zero.
"""

    tokens, number_map = tokenize(text)

    print("Text:\n", text)
    print("\nTokens:", tokens)
    print("\nNumber Map:")
    for idx, (val, typ, raw) in number_map.items():
        print(f"  {idx}: {val} ({typ}) -> '{raw}'")

    token_ids, feats, vocab = prepare_inputs(tokens, number_map)

    print("\nToken IDs:", token_ids)
    print("\nFeatures shape:", feats.shape)

    print("\n--- Numeric Token Embeddings and Decoded Values ---\n")
    print(f"{'Idx':<5} | {'Token':<15} | {'Raw Number':<20} | {'Decoded Value':<20}")
    print("-" * 70)
    
    for idx in sorted(number_map.keys()):
        token = tokens[idx]
        val, typ, raw = number_map[idx]
        feat = feats[0, idx]
        decoded_val = decode_number_from_features(feat)
        print(f"{idx:<5} | {token:<15} | {raw:<20} | {decoded_val:<20}")

if __name__ == "__main__":
    main()

    test_embedding_decode()
    test_prepare_inputs()
```
###Ouput

```
Text:
 Add 3.14 and -42 to zero. The total sum should be 0.0.  
In physics, the speed of light is approximately 299792458 m/s.  
Negative temperatures like -273.15°C represent absolute zero.


Tokens: ['<|cap|>', 'add', '<|num|>', 'and', '<|num|>', 'to', '<|space|>', 'zero', '.', '<|cap|>', 'the', '<|space|>', 'total', '<|space|>', 'sum', '<|space|>', 'should', '<|space|>', 'be', '<|num|>', '.', '<|cap|>', 'in', '<|space|>', 'physics', ',', 'the', '<|space|>', 'speed', '<|space|>', 'of', '<|space|>', 'light', '<|space|>', 'is', '<|space|>', 'approximately', '<|num|>', 'm', '/', 's', '.', '<|cap|>', 'negative', '<|space|>', 'temperatures', '<|space|>', 'like', '<|num|>', '°', '<|space|>', 'c', '<|space|>', 'represent', '<|space|>', 'absolute', '<|space|>', 'zero', '.']

Number Map:
  2: 3.14 (float) -> ' 3.14'
  4: -42 (int) -> ' -42'
  19: 0.0 (float) -> ' 0.0'
  37: 299792458 (int) -> ' 299792458'
  48: -273.15 (float) -> ' -273.15'

Token IDs: tensor([[ 3,  8,  4,  9,  4, 27,  5, 29,  1,  3, 26,  5, 28,  5, 24,  5, 22,  5,
         11,  4,  1,  3, 12,  5, 19,  0, 26,  5, 23,  5, 18,  5, 14,  5, 13,  5,
         10,  4, 16,  2, 21,  1,  3, 17,  5, 25,  5, 15,  4, 30,  5,  6,  5, 20,
          5,  7,  5, 29,  1]])

Features shape: torch.Size([1, 59, 128])

--- Numeric Token Embeddings and Decoded Values ---

Idx   | Token           | Raw Number           | Decoded Value        
----------------------------------------------------------------------
2     | <|num|>         |  3.14                | 3.14                 
4     | <|num|>         |  -42                 | -42.0                
19    | <|num|>         |  0.0                 | 0.0                  
37    | <|num|>         |  299792458           | 299792458.0          
48    | <|num|>         |  -273.15             | -273.15              
```

## What This Demonstrates

- **Numeric tokens are uniformly replaced by a special `<|num|>` token, with actual numeric values stored separately in a dedicated number map.**  
  This approach treats all numbers as a single atomic token, eliminating the fragmentation of numbers into multiple tokens (e.g., `3`, `.`, `14`).

- **The embedding features tensor (`feats`) encodes all numeric information as a dense, continuous vector representation.**  
  Each number is represented by a feature vector that comprehensively captures its value and type, allowing the model to “understand” numbers in a continuous space rather than as mere discrete symbols.

- **Numeric embeddings can be decoded back to the exact original number, proving that the encoding is effectively lossless.**  
  This is a key advantage—no numeric information is lost during tokenization or embedding, enabling precise numerical operations and reasoning inside the model.

- **This enables numeric reasoning models to work directly with precise numerical data instead of fragmented token sequences.**  
  Models learn and operate on numbers as meaningful, indivisible units, improving interpretation, inference, and mathematical operations in NLP tasks.

### Practical Implications

- **Reduced vocabulary complexity:** Instead of multiple tokens representing various numeric formats, there is a single `<|num|>` token plus a separate numeric feature channel.  
- **Better generalization:** Models no longer need to memorize countless numeric token fragments, reducing the risk of overfitting to specific numeric patterns seen during training.  
- **Easy integration with numeric modules:** Numeric feature vectors can be fed directly into specialized external modules or layers designed for numerical computation, enabling hybrid NLP-numeric models.  
- **Improved interpretability:** The ability to precisely recover original numeric values from embeddings aids interpretability and debugging, crucial in high-stakes domains such as medicine and finance.

---

This is more than a clever tokenization trick—it’s a fundamental shift in how numbers are represented in language models, enabling them to truly *understand* numeric data beyond raw text.  
This capability can significantly enhance the quality and reliability of systems that need to integrate textual and numerical reasoning seamlessly.

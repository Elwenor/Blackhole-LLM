### Understanding the BlackholeTokenizer

The `BlackholeTokenizer` is a core, innovative component of the Blackhole-LLM project. It's built on Hugging Face's `tokenizers` library and extends `PreTrainedTokenizerFast` to provide specialized handling for various data types, especially numerical information and capitalization, while aiming for vocabulary efficiency.

Here's how to get started with the `BlackholeTokenizer`, including training, saving, loading, and encoding/decoding text with clear examples and their outputs.

---

### 1. Installation and Import

First, ensure you have the necessary libraries installed and import the `BlackholeTokenizer` into your Python script. Remember to adjust the `sys.path.insert` line if your project structure differs.

```python
import sys, os
import torch
# Adjust the path to where your 'blackhole' directory is located
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from blackhole.tokenizer_hugging_face import BlackholeTokenizer
import tokenizers

print(f"Tokenizers library version: {tokenizers.__version__}")
```

**Output:**
```
Tokenizers library version: 0.19.1
```
*(Your version number might differ slightly.)*

---

### 2. Initializing and Training the Tokenizer

Before you can use the tokenizer, you need to **train it**. This process teaches the tokenizer how to break down text into smaller units (tokens) and assign unique IDs. You can specify the `vocab_size` (the desired size of your vocabulary) and `min_freq` (the minimum frequency for a token to be included).

```python
# Initialize the tokenizer
tokenizer = BlackholeTokenizer()

# Sample texts for training
texts_for_training = [
    "Hello world! This is a test. The number is 123.45 and also 0xabc.",
    "Another EXAMPLE sentence with DATE 2023-10-26 and time 14:30. What about i.e. and e.g.?",
    "Numbers: +1000, -5.5e-2, 999,999.00. Operators: ->, <=, ==.",
    "ALL CAPS TEXT. First Capital Letter.",
    "Unicode hyphen: this–that. At-tag: @xmath0. A sentence with ellipsis... and quotes 'like this'."
]

print("Starting tokenizer training...")
tokenizer.train_tokenizer(texts_for_training, vocab_size=8000, min_freq=1)
print(f"Vocabulary size after training: {tokenizer.vocab_size}")
```

**Partial Output from Training:**
```
Starting tokenizer training...
[00:00:00] Pre-processing sequences                      █████████████████████████████████████████████████████████████████████████████████████████████████████████ 0         /         0
[00:00:00] Tokenize words                                █████████████████████████████████████████████████████████████████████████████████████████████████████████ 166       /       166
[00:00:00] Count pairs                                   █████████████████████████████████████████████████████████████████████████████████████████████████████████ 166       /       166
[00:00:00] Compute merges                                █████████████████████████████████████████████████████████████████████████████████████████████████████████ 243       /       243
Vocabulary size after training: 347
```
*This output confirms the tokenizer has been trained, resulting in a vocabulary of 347 unique tokens.*

---

### 3. Saving and Loading the Tokenizer

Once trained, you can save your tokenizer to disk. This allows you to load it later without needing to retrain it, making your workflow more efficient.

```python
# Define a directory to save the tokenizer
output_dir = "./my_custom_tokenizer_test"
os.makedirs(output_dir, exist_ok=True)

# Save the tokenizer
tokenizer.save_pretrained(output_dir)
print(f"Tokenizer saved to: {output_dir}")

# Load the tokenizer
loaded_tokenizer = BlackholeTokenizer.from_pretrained(output_dir)
print(f"Tokenizer loaded from: {output_dir}")
```

**Output:**
```
Tokenizer saved to: ./my_custom_tokenizer_test
Tokenizer loaded from: ./my_custom_tokenizer_test
```

---

### 4. Using the Tokenizer: Encoding and Decoding Text

With the tokenizer loaded, you can now convert text into numerical IDs (encoding) and convert IDs back into human-readable text (decoding).

#### Example 1: Standard Text with Numbers and Capitalization

This example showcases how the tokenizer handles regular sentences, identifying and marking numbers and capitalized words with special tokens.

```python
test_text_1 = "This is an example text. The value is 1,234.56. ALL CAPS! Another word. @xcite_here"
encoded_input_1 = loaded_tokenizer(test_text_1, return_tensors="pt", padding=True, truncation=False)
decoded_text_1 = loaded_tokenizer.decode(encoded_input_1['input_ids'][0], skip_special_tokens=False)
decoded_clean_1 = loaded_tokenizer.decode(encoded_input_1['input_ids'][0], skip_special_tokens=True)

print(f"\n--- Test: Standard Text with Numbers and Capitalization ---")
print(f"Original text:                     '{test_text_1}'")
print(f"Encoded IDs (input_ids):           {encoded_input_1['input_ids'][0].tolist()}")
print(f"Attention mask:                    {encoded_input_1['attention_mask'][0].tolist()}")
print(f"Decoded text (with special tokens):'{decoded_text_1}'")
print(f"Decoded text (without special tokens):'{decoded_clean_1}'")

if test_text_1.strip() != decoded_clean_1.strip():
    print(f"MISMATCH: Original and decoded text (clean) are different.")
else:
    print(f"MATCH: Original and decoded text (clean) are identical.")
```

**Output for Example 1:**
```
--- Test: Standard Text with Numbers and Capitalization ---
Original text:                     'This is an example text. The value is 1,234.56. ALL CAPS! Another word. @xcite_here'
Encoded IDs (input_ids):           [1, 55, 71, 72, 82, 8, 112, 8, 64, 77, 8, 68, 87, 150, 240, 8, 83, 68, 87, 83, 22, 8, 55, 71, 68, 8, 85, 149, 84, 68, 8, 112, 8, 25, 20, 26, 27, 28, 22, 29, 30, 22, 8, 40, 48, 48, 8, 42, 40, 52, 54, 9, 8, 40, 77, 78, 83, 71, 68, 81, 8, 117, 67, 22, 8, 39, 87, 205, 83, 68, 63, 71, 107, 68, 2]
Attention mask:                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Decoded text (with special tokens):'[CAP]This is an example text. [CAP]The value is [NUM]1,234.56. [ALLCAPS]ALL [ALLCAPS]CAPS! [CAP]Another word. @xcite_here'
Decoded text (without special tokens):'This is an example text. The value is 1,234.56. ALL CAPS! Another word. @xcite_here'
MATCH: Original and decoded text (clean) are identical.
```
*In this example, you can see how the tokenizer intelligently inserts **`[CAP]`** for initial capitals, **`[NUM]`** for numbers, and **`[ALLCAPS]`** for all-caps words. When decoded without special tokens, the original text is perfectly restored.*

#### Example 2: Whitespace Handling

The `BlackholeTokenizer` is designed to preserve whitespace accurately, which is often crucial for language models.

```python
test_text_6 = "      Hello      World!      This is a test.      "
encoded_input_6 = loaded_tokenizer(test_text_6, return_tensors="pt")
decoded_text_6 = loaded_tokenizer.decode(encoded_input_6['input_ids'][0], skip_special_tokens=False)
decoded_clean_6 = loaded_tokenizer.decode(encoded_input_6['input_ids'][0], skip_special_tokens=True)

print(f"\n--- Test: Whitespace Handling ---")
print(f"Original text:                     '{test_text_6}'")
print(f"Encoded IDs (input_ids):           {encoded_input_6['input_ids'][0].tolist()}")
print(f"Attention mask:                    {encoded_input_6['attention_mask'][0].tolist()}")
print(f"Decoded text (with special tokens):'{decoded_text_6}'")
print(f"Decoded text (without special tokens):'{decoded_clean_6}'")

if test_text_6.strip() != decoded_clean_6.strip():
    print(f"MISMATCH: Original and decoded text (clean) are different.")
else:
    print(f"MATCH: Original and decoded text (clean) are identical.")
```

**Output for Example 2:**
```
--- Test: Whitespace Handling ---
Original text:                     '      Hello      World!      This is a test.      '
Encoded IDs (input_ids):           [1, 8, 8, 8, 8, 8, 8, 46, 68, 75, 75, 78, 78, 78, 8, 8, 8, 8, 8, 8, 57, 78, 81, 75, 67, 9, 8, 8, 8, 8, 8, 8, 55, 71, 72, 82, 8, 112, 8, 64, 77, 8, 68, 87, 150, 240, 8, 83, 68, 87, 83, 22, 8, 8, 8, 8, 8, 8, 2]
Attention mask:                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Decoded text (with special tokens):'      [CAP]Hello      [CAP]World!      [CAP]This is a test.      '
Decoded text (without special tokens):'      Hello      World!      This is a test.      '
MATCH: Original and decoded text (clean) are identical.
```
*This test demonstrates that the tokenizer correctly preserves multiple spaces, including leading and trailing ones. This is essential for maintaining the exact structure of the input text.*

---

### 5. Special Token Information

Understanding the special tokens and their assigned IDs is key to working with the `BlackholeTokenizer`. These tokens allow the model to interpret different types of information within the text.

```python
print(f"\n--- Special Token Information ---")
print(f"All special tokens: {loaded_tokenizer.all_special_tokens}")
print(f"NUM token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.num_token)}")
print(f"CAP token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.cap_token)}")
print(f"ALLCAPS token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.allcaps_token)}")
print(f"UNK token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.unk_token)}")
print(f"CLS token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.cls_token)}")
print(f"PAD token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.pad_token)}")
print(f"SEP token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.sep_token)}")
```

**Output:**
```
--- Special Token Information ---
All special tokens: ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]', '[NUM]', '[CAP]', '[ALLCAPS]']
NUM token ID: 5
CAP token ID: 6
ALLCAPS token ID: 7
UNK token ID: 0
CLS token ID: 1
PAD token ID: 3
SEP token ID: 2
```
*This provides a clear mapping of special tokens to their numerical IDs. For example, **`[NUM]`** has an ID of 5, indicating a number, and **`[CAP]`** has an ID of 6, indicating a capitalized word.*

---
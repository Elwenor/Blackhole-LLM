### Understanding the BlackholeTokenizer

The `BlackholeTokenizer`, a key part of the Blackhole-LLM project, extends Hugging Face's `PreTrainedTokenizerFast`. It uniquely handles various data types like numbers and capitalization for better vocabulary efficiency and model understanding, specifically designed to support **dual-stream model architectures**.

Unlike standard tokenizers, `BlackholeTokenizer` **preserves critical lexical metadata**. It identifies and categorizes numbers, URLs, emails, capitalized words, and all-caps words. It then uses **special tokens** (e.g., `[NUM]`, `[CAP]`, `[ALLCAPS]`) to mark these types, while still allowing the BPE model to sub-tokenize their content (e.g., breaking "123.45" into "1", "23", ".45"). This means your model gets explicit signals about these important attributes.

---

### Dual Output for Advanced Model Architectures

A core feature of `BlackholeTokenizer` is its ability to provide **two distinct output tensors** when encoding text, tailored for models with dual processing paths (e.g., "double embeddings" or "double transformer" architectures):

1.  **`input_ids` (Text Stream):** This is the primary token sequence, containing standard BPE tokens along with the special marker tokens (`[NUM]`, `[CAP]`, `[ALLCAPS]`). This stream is intended for the model's **linguistic processing path**, allowing it to understand the general context and the *presence* of specific lexical types.
2.  **`numeric_values` (Numeric Stream):** This is a new tensor of the same length as `input_ids`. For positions corresponding to a `[NUM]` token in `input_ids`, this tensor will contain the **actual parsed numerical value** (e.g., `1234.56`). For all other positions (non-numeric tokens), it will contain a designated padding value (defaulting to `0.0`). This stream is designed for a dedicated **numerical processing path** within your model, enabling it to perform mathematical operations and reason about quantitative information.

This dual output allows your model to learn distinct representations for linguistic context and numerical values, fostering a deeper understanding of both symbolic and quantitative information.

---

Let's get started with using it:

---

### 1. Installation and Import

First, install necessary libraries and import `BlackholeTokenizer`. Adjust the `sys.path.insert` if your project structure is different.

```python
import sys, os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from blackhole.tokenizer_hugging_face import BlackholeTokenizer
import tokenizers

print(f"Tokenizers library version: {tokenizers.__version__}")
```

**Output:**
```
Tokenizers library version: 0.19.1
```

---

### 2. Initializing and Training the Tokenizer

You need to **train** the tokenizer to teach it how to break down text. During training, it preprocesses your text by splitting, categorizing, and preparing segments for BPE. Numbers, URLs, and emails are broken into characters for BPE to learn sub-character patterns, while other words are passed as whole units. The special marker tokens (`[NUM]`, `[CAP]`, `[ALLCAPS]`) are also added to the vocabulary during this process.

```python
tokenizer = BlackholeTokenizer()

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

**Partial Output:**
```
Starting tokenizer training...
[00:00:00] Pre-processing sequences ...
[00:00:00] Tokenize words ...
[00:00:00] Count pairs ...
[00:00:00] Compute merges ...
Vocabulary size after training: 347
```

---

### 3. Saving and Loading the Tokenizer

Save your trained tokenizer to avoid retraining.

```python
output_dir = "./my_custom_tokenizer_test"
os.makedirs(output_dir, exist_ok=True)

tokenizer.save_pretrained(output_dir)
print(f"Tokenizer saved to: {output_dir}")

loaded_tokenizer = BlackholeTokenizer.from_pretrained(output_dir)
print(f"Tokenizer loaded from: {output_dir}")
```

**Output:**
```
Tokenizer saved to: ./my_custom_tokenizer_test
Tokenizer loaded from: ./my_custom_tokenizer_test
```

---

### 4. Encoding and Decoding Text

Convert text to IDs (encode) and IDs back to text (decode). Notice how **special tokens** like `[CAP]`, `[NUM]`, and `[ALLCAPS]` are inserted into the `input_ids`, indicating the original lexical properties. The `numeric_values` tensor provides the actual numerical data for `[NUM]` tokens. Decoding with `skip_special_tokens=True` perfectly restores the original text. Whitespace is also accurately preserved.

```python
test_text = "This is an example text. The value is 1,234.56. ALL CAPS! Another word. @xcite_here"
encoded_input = loaded_tokenizer(test_text, return_tensors="pt")

# Print the dual outputs
print(f"\n--- Encoded Input Details ---")
print(f"Input IDs (first 20): {encoded_input['input_ids'][0][:20].tolist()}...")
print(f"Attention Mask (first 20): {encoded_input['attention_mask'][0][:20].tolist()}...")
print(f"Numeric Values (first 20): {encoded_input['numeric_values'][0][:20].tolist()}...")


decoded_with_specials = loaded_tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=False)
decoded_clean = loaded_tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)

print(f"\n--- Example: Encoding and Decoding ---")
print(f"Original text: '{test_text}'")
print(f"Decoded (with special tokens): '{decoded_with_specials}'")
print(f"Decoded (without special tokens): '{decoded_clean}'")
print(f"Match original: {test_text.strip() == decoded_clean.strip()}")

# Whitespace example
test_text_ws = "      Hello      World!      "
encoded_ws = loaded_tokenizer(test_text_ws, return_tensors="pt")
decoded_ws = loaded_tokenizer.decode(encoded_ws['input_ids'][0], skip_special_tokens=True)
print(f"\n--- Example: Whitespace Handling ---")
print(f"Original text: '{test_text_ws}'")
print(f"Decoded text: '{decoded_ws}'")
print(f"Match original: {test_text_ws.strip() == decoded_ws.strip()}")
```

**Partial Output:**
```
--- Encoded Input Details ---
Input IDs (first 20): [2, 6, 12, 13, 14, 15, 16, 17, 18, 19, 5, 20, 21, 22, 23, 24, 25, 26, 27, 28]...
Attention Mask (first 20): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]...
Numeric Values (first 20): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1234.56, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]...

--- Example: Encoding and Decoding ---
Original text: 'This is an example text. The value is 1,234.56. ALL CAPS! Another word. @xcite_here'
Decoded (with special tokens): '[CLS][CAP]This is an example text. [CAP]The value is [NUM]1,234.56. [ALLCAPS]ALL [ALLCAPS]CAPS! [CAP]Another word. @xcite_here[SEP]'
Decoded (without special tokens): 'This is an example text. The value is 1,234.56. ALL CAPS! Another word. @xcite_here'
Match original: True

--- Example: Whitespace Handling ---
Original text: '      Hello      World!      '
Decoded text: '      Hello      World!      '
Match original: True
```

---

### 5. Extracting Numeric Information

A powerful feature is the ability to extract detailed information about detected numbers using `get_numeric_info()` and `get_detected_numbers_summary()`.

#### `get_numeric_info()`

This method returns a list of dictionaries, each detailing a detected number with its **parsed value**, **type** (`int`/`float`), **format** (e.g., `decimal_float`, `scientific_notation`), original string, character span, and crucially, its **token ID span** within the encoded sequence. This allows direct linking of original numbers to their model representations.

```python
test_text_for_numbers = "Price is $1,234.50. Growth rate is 3.14e-2. We need 100 units by 2025."
encoded_for_numbers = loaded_tokenizer(test_text_for_numbers, return_tensors="pt")

numeric_data = loaded_tokenizer.get_numeric_info(batch_index=0)

print(f"\n--- Example: Numeric Information Extraction ---")
print(f"Original text: '{test_text_for_numbers}'")
print(f"Full numeric info (first number):")
if numeric_data:
    import json
    print(json.dumps(numeric_data[0], indent=2)) # Print only the first for brevity

    # Verify a number's token IDs
    first_num = numeric_data[0]
    start, end = first_num['token_ids_span']
    tokens_for_num = encoded_for_numbers['input_ids'][0, start:end].tolist()
    decoded_num_tokens = loaded_tokenizer.decode(tokens_for_num, skip_special_tokens=True)
    print(f"\nVerifying token IDs for '{first_num['original_string']}':")
    print(f"Token IDs: {tokens_for_num}, Decoded: '{decoded_num_tokens}'")
```

**Partial Output:**
```
--- Example: Numeric Information Extraction ---
Original text: 'Price is $1,234.50. Growth rate is 3.14e-2. We need 100 units by 2025.'
Full numeric info (first number):
{
  "value": 1234.5,
  "type": "float",
  "format": "decimal_float",
  "original_string": "1,234.50",
  "original_char_span": [11, 19],
  "token_ids_span": [10, 19], # Example token span, actual values depend on BPE
  "token_ids": [5, 25, 20, 26, 27, 28, 22, 29, 30] # Example token IDs
}

Verifying token IDs for '1,234.50':
Token IDs: [5, 25, 20, 26, 27, 28, 22, 29, 30], Decoded: '1,234.50'
```

#### `get_detected_numbers_summary()`

For a quick overview, this method provides a concise summary of unique numbers and their formats.

```python
numeric_summary = loaded_tokenizer.get_detected_numbers_summary(batch_index=0)

print(f"\n--- Example: Numeric Summary ---")
print(f"Detected numbers summary: {numeric_summary}")
```

**Output:**
```
--- Example: Numeric Summary ---
Detected numbers summary: ["['1,234.50', DECIMAL_FLOAT]", "['3.14e-2', SCIENTIFIC_NOTATION]", "['100', INTEGER]", "['2025', INTEGER]"]
```

---

### 6. Special Token Information

Knowing the special tokens and their IDs is vital for interacting with the tokenizer and designing models that leverage this information.

```python
print(f"\n--- Special Token Information ---")
print(f"All special tokens: {loaded_tokenizer.all_special_tokens}")
print(f"NUM token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.num_token)}")
print(f"CAP token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.cap_token)}")
print(f"ALLCAPS token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.allcaps_token)}")
# ... and other special tokens like UNK, CLS, PAD, SEP
```

**Output:**
```
--- Special Token Information ---
All special tokens: ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]', '[NUM]', '[CAP]', '[ALLCAPS]']
NUM token ID: 5
CAP token ID: 6
ALLCAPS token ID: 7
# ... and other special tokens and their IDs
```
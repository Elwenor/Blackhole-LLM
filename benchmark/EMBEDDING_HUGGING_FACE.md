## Integrating Blackhole into Your LLM Training: Modularized Examples

Incorporating Blackhole into your LLM training pipeline involves two main components: **input tokenization** and **leveraging a custom embedding layer**. This updated guide breaks down the usage of `BlackholeTokenizer` and `BlackholeEmbeddings` into more digestible and independently testable code blocks.

---

### 1. Preparing Input Data with `BlackholeTokenizer`

The `BlackholeTokenizer` is crucial for preprocessing raw text into a format suitable for your Blackhole embedding layer. It handles standard tokenization, detects and replaces numbers with a special `[NUM]` token, and extracts numerical metadata (value and format).

#### 1.1. `BlackholeTokenizer` Initialization and Training

First, you'll initialize and train the tokenizer using a representative corpus.

```python
import sys
import os
import torch
from blackhole.tokenizer_hugging_face import BlackholeTokenizer, CUSTOM_SPECIAL_TOKENS

# Add paths to allow Python to find your modules
# Assuming 'test_blackhole_embeddings.py' is in the project root,
# and 'blackhole_embeddings.py' and the 'blackhole' directory are alongside it.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

print("--- 1.1: Initializing and Training BlackholeTokenizer ---")

# Initialize and train the tokenizer
tokenizer = BlackholeTokenizer()
sample_texts_for_training = [
    "The temperature is 25.5 degrees Celsius.",
    "My bank balance is -123.45 dollars.",
    "The population is 8.0e9 people.",
    "A simple integer: 42.",
    "The number of users: 1000.",
    "It costs 99.99 EUR.",
    "No numbers here."
]
tokenizer.train_tokenizer(sample_texts_for_training, vocab_size=8000, min_freq=1)

# Get the ID of the special [NUM] token
num_token_id = tokenizer.vocab.get(CUSTOM_SPECIAL_TOKENS["number_token"])
if num_token_id is None:
    raise ValueError(f"Token '{CUSTOM_SPECIAL_TOKENS['number_token']}' not found in vocabulary. Ensure the tokenizer was trained correctly.")

print(f"Tokenizer trained successfully. Vocab size: {tokenizer.vocab_size}")
print(f"ID for special [NUM] token: {num_token_id}")

# Save and load tokenizer to demonstrate persistence
output_dir = "./my_blackhole_tokenizer_test"
os.makedirs(output_dir, exist_ok=True)
tokenizer.save_pretrained(output_dir)
print(f"Tokenizer saved to {output_dir}")
loaded_tokenizer = BlackholeTokenizer.from_pretrained(output_dir)
print(f"Tokenizer loaded from {output_dir}")

```

#### 1.2. `BlackholeTokenizer` Encoding Examples

Now, let's see how the tokenizer processes different types of text, including numbers, special characters, and various formatting.

```python
import torch
from typing import List, Dict, Any, Tuple, Optional, Union

# Assuming loaded_tokenizer is available from the previous step
# If running this section independently, uncomment and run the '1.1' block first.
# from blackhole.tokenizer_hugging_face import BlackholeTokenizer
# loaded_tokenizer = BlackholeTokenizer.from_pretrained("./my_blackhole_tokenizer_test") # Load if not already in memory

def print_tokenizer_test_results(title, original_text, encoded_output, tokenizer_obj):
    """Helper function to print tokenizer test results."""
    print(f"\n--- Tokenizer Test: {title} ---")
    print(f"Original text:                 '{original_text}'")

    # Normalize input_ids, numeric_values, numeric_formats, and attention_mask to flat lists
    input_ids_list = encoded_output['input_ids'][0].tolist() if isinstance(encoded_output['input_ids'], torch.Tensor) else encoded_output['input_ids'][0]
    numeric_values_list = encoded_output['numeric_values'][0].tolist() if isinstance(encoded_output['numeric_values'], torch.Tensor) else encoded_output['numeric_values'][0]
    numeric_formats_list = encoded_output['numeric_formats'][0].tolist() if isinstance(encoded_output['numeric_formats'], torch.Tensor) else encoded_output['numeric_formats'][0]
    attention_mask_list = encoded_output['attention_mask'][0].tolist() if isinstance(encoded_output['attention_mask'], torch.Tensor) else encoded_output['attention_mask'][0]

    # Convert IDs to string tokens
    encoded_tokens = tokenizer_obj.convert_ids_to_tokens(input_ids_list)

    # Decode with and without special tokens
    decoded_with_special = tokenizer_obj.decode(input_ids_list, skip_special_tokens=False)
    decoded_clean = tokenizer_obj.decode(input_ids_list, skip_special_tokens=True)

    print(f"Encoded input (IDs):           {input_ids_list}")
    print(f"Encoded tokens (from IDs):     {encoded_tokens}")
    print(f"Numeric Values:                {numeric_values_list}")
    print(f"Numeric Formats:               {numeric_formats_list}")
    print(f"Attention mask:                {attention_mask_list}")
    print(f"Decoded text (with special):   '{decoded_with_special}'")
    print(f"Decoded text (without special):'{decoded_clean}'")

    # Verification
    if original_text.strip() != decoded_clean.strip():
        print(f"MISMATCH: Original and decoded text (clean) are different.")
    else:
        print(f"MATCH: Original and decoded text (clean) are identical.")

    # Check for [NUM] token and its associated data
    num_token_id = tokenizer_obj.vocab.get(CUSTOM_SPECIAL_TOKENS["number_token"])
    num_token_indices = [i for i, id_val in enumerate(input_ids_list) if id_val == num_token_id]

    if num_token_indices:
        print(f"Found [NUM] token at indices: {num_token_indices}")
        for idx in num_token_indices:
            val = numeric_values_list[idx]
            fmt = numeric_formats_list[idx]
            print(f"  - At index {idx}: Value={val}, Format={fmt}")
    else:
        print("No [NUM] token found in this encoding.")


print("\n--- 1.2: BlackholeTokenizer Encoding Examples ---")

# Example 1: Basic sentence with a float
text_1 = "The price is 12.99 USD."
encoded_1 = loaded_tokenizer(text_1, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
print_tokenizer_test_results("Basic sentence with float", text_1, encoded_1, loaded_tokenizer)

# Example 2: Negative integer
text_2 = "The absolute value is -50.0."
encoded_2 = loaded_tokenizer(text_2, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
print_tokenizer_test_results("Negative integer", text_2, encoded_2, loaded_tokenizer)

# Example 3: Scientific notation
text_3 = "Earth's population is approximately 8.0e9 people."
encoded_3 = loaded_tokenizer(text_3, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
print_tokenizer_test_results("Scientific notation", text_3, encoded_3, loaded_tokenizer)

# Example 4: Text without numbers
text_4 = "This sentence contains no numbers."
encoded_4 = loaded_tokenizer(text_4, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
print_tokenizer_test_results("Text without numbers", text_4, encoded_4, loaded_tokenizer)

# Example 5: Multiple numbers in one sentence
text_5 = "Item A costs 5.50, Item B costs 10.0, and there are 3 of them."
encoded_5 = loaded_tokenizer(text_5, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
print_tokenizer_test_results("Multiple numbers", text_5, encoded_5, loaded_tokenizer)
```

---

### 2. Replacing the Standard Embedding Layer with `BlackholeEmbeddings`

Instead of a standard `nn.Embedding` layer, you'll use `BlackholeEmbeddings` to intelligently combine textual information with rich numerical features. This layer takes the `input_ids`, `numeric_values`, and `numeric_formats` tensors generated by the tokenizer.

#### 2.1. `BlackholeEmbeddings` Configuration and Initialization

First, define the configuration for `BlackholeEmbeddings` and initialize the layer.

```python
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig
from blackhole.embadding_hugging_face import BlackholeEmbeddings, BlackholeConfig

# Assuming num_token_id and tokenizer are available from previous steps.
# If running this section independently, make sure to define them:
# num_token_id = <your_num_token_id_from_tokenizer>
# tokenizer = BlackholeTokenizer.from_pretrained("./my_blackhole_tokenizer_test")


print("\n--- 2.1: BlackholeEmbeddings Configuration and Initialization ---")

# Define the configuration for Blackhole Embeddings
# Ensure `num_token_id` matches the one from your tokenizer!
bh_config = BlackholeConfig(
    vocab_size=tokenizer.vocab_size, # From your tokenizer
    hidden_size=256,                 # Example: embedding dimension for LLMs
    max_position_embeddings=128,     # Maximum sequence length
    pad_token_id=tokenizer.pad_token_id,
    num_token_id=num_token_id,       # [NUM] token ID from the tokenizer
    numeric_feature_dims={           # Configuration for numerical features
        "log_value": 1,
        "sign": 1,
        "exponent": 1,
        "binary_representation": 16, # Example: 16 "bits" for simplified binary representation
        "format_type": 3,            # Example: 3 format types (0:int, 1:float, 2:scientific)
    },
    numeric_embedding_fusion_type="gating", # Choose your preferred fusion type: "gating", "add", or "concat"
    # Add other standard LLM configuration parameters here if needed
)

# Initialize the embeddings layer
embeddings_layer = BlackholeEmbeddings(bh_config)
print(f"BlackholeEmbeddings initialized successfully with fusion type: '{bh_config.numeric_embedding_fusion_type}'")
print(f"Expected embedding output size: {bh_config.hidden_size}")
```

#### 2.2. `BlackholeEmbeddings` Usage Examples

Here, we'll demonstrate how to pass tokenized data to `BlackholeEmbeddings` and verify the output.

```python
import torch
import numpy as np # For np.isclose and np.nan

# Assuming embeddings_layer, tokenizer, and bh_config are available from previous steps.
# If running this section independently, make sure to define them.

def verify_embedding_output(
    text: str,
    expected_num_value: Optional[float], # Use Optional as not all texts have numbers
    expected_format_id: Optional[int],   # 0: int, 1: float, 2: scientific, -1: non-numeric
    embeddings_layer: BlackholeEmbeddings,
    tokenizer: BlackholeTokenizer,
    config: BlackholeConfig,
    description: str
):
    """Helper function to verify the output of BlackholeEmbeddings."""
    print(f"\n--- BlackholeEmbeddings Test for: {description} ---")
    print(f"Text: '{text}'")

    encoded_input = tokenizer(
        text,
        padding="max_length",
        max_length=config.max_position_embeddings,
        return_tensors="pt"
    )

    input_ids = encoded_input['input_ids']
    numeric_values = encoded_input['numeric_values']
    numeric_formats = encoded_input['numeric_formats']

    print(f"Input IDs (first 10): {input_ids[0, :10].tolist()}")
    print(f"Numeric Values (first 10): {numeric_values[0, :10].tolist()}")
    print(f"Numeric Formats (first 10): {numeric_formats[0, :10].tolist()}")

    # Pass to embeddings_layer
    final_embeddings = embeddings_layer(
        input_ids=input_ids,
        numeric_values=numeric_values,
        numeric_formats=numeric_formats
    )

    print(f"Shape of final embeddings: {final_embeddings.shape}")
    assert final_embeddings.shape == (1, config.max_position_embeddings, config.hidden_size), \
        f"Mismatch in final embedding shape! Expected {(1, config.max_position_embeddings, config.hidden_size)}, got {final_embeddings.shape}"

    # Find the position of the [NUM] token
    num_token_pos = (input_ids == config.num_token_id).nonzero(as_tuple=True)[1]

    if num_token_pos.numel() > 0:
        pos = num_token_pos[0].item() # Take the first instance
        
        # Verify the numeric_values and numeric_formats at the [NUM] token position
        actual_val = numeric_values[0, pos].item()
        actual_format = numeric_formats[0, pos].item()

        print(f"Found [NUM] token at position {pos}. Token value extracted: {actual_val}, Format extracted: {actual_format}")
        
        if expected_num_value is not None:
            # Handle potential NaN from tokenizer if the input wasn't numeric
            if np.isnan(expected_num_value) and np.isnan(actual_val):
                print("Confirmed NaN value for non-numeric token (expected).")
            elif not np.isclose(actual_val, expected_num_value) or actual_format != expected_format_id:
                print(f"ERROR: Expected value/format ({expected_num_value}/{expected_format_id}) "
                      f"does not match received ({actual_val}/{actual_format}). Check tokenizer output!")
                return False

        # Get the embedding for the [NUM] token
        num_embedding = final_embeddings[0, pos]
        print(f"Fragment of embedding for [NUM] (first 5 dimensions): {num_embedding[:5].tolist()}")
        
        # Check if the embedding is meaningful (not all zeros)
        is_meaningful = not torch.allclose(num_embedding, torch.zeros_like(num_embedding), atol=1e-5)
        print(f"Is [NUM] embedding meaningful (not all zeros)? {is_meaningful}")
        assert is_meaningful, f"Embedding for [NUM] token ({expected_num_value}) is all zeros!"
        print("Test successful: [NUM] token processed correctly.")
    else:
        print("No [NUM] token expected or found in this text. Checking general embedding integrity.")
        # For non-numeric texts, ensure general embeddings are still meaningful
        is_any_embedding_meaningful = False
        for i in range(input_ids.size(1)):
            if input_ids[0, i].item() != config.pad_token_id:
                if not torch.allclose(final_embeddings[0, i], torch.zeros_like(final_embeddings[0, i]), atol=1e-5):
                    is_any_embedding_meaningful = True
                    break
        assert is_any_embedding_meaningful, "All embeddings are zeros even for non-padding tokens in a text without numbers!"
        print("Test successful: General embeddings are meaningful for text without numbers.")

# --- Running Key Scenarios for BlackholeEmbeddings ---
print("\n" + "="*50)
print("--- RUNNING BLACKHOLE EMBEDDINGS KEY SCENARIOS ---")
print("="*50)

# Test 1: Positive floating-point number
verify_embedding_output(
    text="The value is 123.45.",
    expected_num_value=123.45,
    expected_format_id=1, # float
    embeddings_layer=embeddings_layer,
    tokenizer=loaded_tokenizer, # Use loaded_tokenizer from previous step
    config=bh_config,
    description="Positive floating-point number"
)

# Test 2: Negative integer
verify_embedding_output(
    text="Temperature dropped to -10 degrees.",
    expected_num_value=-10.0,
    expected_format_id=0, # int
    embeddings_layer=embeddings_layer,
    tokenizer=loaded_tokenizer,
    config=bh_config,
    description="Negative integer"
)

# Test 3: Number in scientific notation
verify_embedding_output(
    text="Mass of electron is 9.109e-31 kg.",
    expected_num_value=9.109e-31,
    expected_format_id=2, # scientific
    embeddings_layer=embeddings_layer,
    tokenizer=loaded_tokenizer,
    config=bh_config,
    description="Number in scientific notation"
)

# Test 4: Zero
verify_embedding_output(
    text="Zero profit: 0.",
    expected_num_value=0.0,
    expected_format_id=0, # int
    embeddings_layer=embeddings_layer,
    tokenizer=loaded_tokenizer,
    config=bh_config,
    description="Zero value"
)

# Test 5: Text without numbers (should not activate numerical embedding path)
verify_embedding_output(
    text="This sentence has no numbers.",
    expected_num_value=np.nan, # Expect NaN as no number is present
    expected_format_id=-1,     # Expect -1 as no number is present
    embeddings_layer=embeddings_layer,
    tokenizer=loaded_tokenizer,
    config=bh_config,
    description="Text without numbers"
)

# Test 6: Multiple numbers in a sentence
verify_embedding_output(
    text="There are 5 apples and 10 oranges.",
    expected_num_value=5.0, # Will only check the first detected number's position
    expected_format_id=0,
    embeddings_layer=embeddings_layer,
    tokenizer=loaded_tokenizer,
    config=bh_config,
    description="Multiple numbers in a sentence"
)
```

---

### Why Blackhole for Your LLM?

Integrating Blackhole into your LLM is a qualitative leap in its capabilities, enabling a **deeper understanding and more accurate handling of numerical information**. This leads to:

* **Deeper Understanding of Numbers**: Your model will perceive "123" and "456" not just as distinct tokens, but as positive integers within a specific magnitude range, crucial for differentiating values like "10" vs. "1000."
* **Enhanced Quantitative Reasoning**: The model gains the ability to perform basic arithmetic, compare values, and identify numerical trends, improving its answers to questions like "How much would five of these products cost if one is 12.99?"
* **More Accurate Text Generation**: When generating numbers, the model acts with greater awareness of value and context, reducing errors like "Earth has 800 inhabitants" instead of "8 billion."
* **Improved Domain-Specific Performance**: In fields like finance, science, or medicine, where numbers are critical, the model becomes significantly better at processing specialized data.
* **Reduced Numerical Hallucinations**: A better understanding of numbers directly translates to fewer instances of the model "hallucinating" false numerical data.

By implementing `BlackholeTokenizer` and `BlackholeEmbeddings`, you provide your LLM with a robust foundation for **truly understanding and interacting with the world of numbers**, transcending mere textual representation. This significantly enhances your model's power and versatility.
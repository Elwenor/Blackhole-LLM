# test_nova_architecture.py
import os
import sys
import torch
import numpy as np
import shutil
from pathlib import Path

# Adjust the path to import from the blackhole package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..\..')))

from blackhole.nova_hugging_face_encoder import *
from blackhole.nova_hugging_face_encoder.configuration_nova import BlackholeConfig

from blackhole.embadding_hugging_face import BlackholeEmbeddings, BlackholeConfig
from blackhole.tokenizer_hugging_face import BlackholeTokenizer, CUSTOM_SPECIAL_TOKENS

# --- Helper function to set up the tokenizer ---
def setup_tokenizer(output_dir="./blackhole_tokenizer_demo"):
    """Initializes and trains BlackholeTokenizer, then saves and loads it."""
    print("\n" + "="*80)
    print("--- 1. Blackhole Tokenizer Configuration and Initialization ---".center(80))
    print("="*80)

    tokenizer = BlackholeTokenizer()
    sample_texts_for_training = [
    "The temperature is 25.5 degrees Celsius.",
    "My bank account balance is -123.45 dollars.",
    "The global population is approximately 8.0e9 people.",
    "An integer: 42.",
    "User count: 1000.",
    "Price tag: 99.99 EUR.",
    "This sentence has no numbers.",
    "The value is 0xAF and 0b101.",
    "It's about -3.14 degrees.",
    "Battery voltage is 3.7 volts.",
    "There are 365 days in a year.",
    "Pi is approximately 3.14159.",
    "Speed: 88.8 km/h.",
    "Memory usage: 4096 MB.",
    "The hex code is 0x1F4.",
    "Binary switch state: 0b1101.",
    "Altitude is 8848 meters.",
    "Weight of object: 72.5 kg.",
    "Area: 55.55 square meters.",
    "Distance: -42.42 kilometers.",
    "Current draw: 0.02 amperes.",
    "Threshold is set at -0.001.",
    "The value is 2.71828, also known as e.",
    "There were 404 errors.",
    "Time elapsed: 12.34 seconds.",
    "Temperature dropped to -15.0 Celsius.",
    "Profit margin is 33.3 percent.",
    "Zoom level: 150%.",
    "Disk space left: 512.0 GB.",
    "Pressure: 1013.25 hPa.",
    "Random float: 0.123456.",
    "Max depth reached: 10994 meters.",
    "Heart rate: 72 bpm.",
    "Audio bit rate: 320 kbps.",
    "Light intensity: 700 lux.",
    "Radiation dose: 0.5 Sv.",
    "IQ score: 130.",
    "Percentage: 99.99%",
    "Hex color: 0xFF5733."
    ]

    print(f"Training tokenizer on {len(sample_texts_for_training)} example sentences...")
    tokenizer.train_tokenizer(sample_texts_for_training, vocab_size=8000, min_freq=1)

    num_token_id = tokenizer.vocab.get(CUSTOM_SPECIAL_TOKENS["number_token"])
    if num_token_id is None:
        raise ValueError(f"Token '{CUSTOM_SPECIAL_TOKENS['number_token']}' not found in vocabulary. Ensure it was added during training.")
    print(f"-> ID for special token {CUSTOM_SPECIAL_TOKENS['number_token']}: {num_token_id}")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    loaded_tokenizer = BlackholeTokenizer.from_pretrained(output_dir)
    print(f"-> Tokenizer loaded successfully from '{output_dir}'")
    return loaded_tokenizer, num_token_id, output_dir

# --- Function to test the embedding layer ---
def test_embedding_layer(tokenizer, num_token_id, freeze_heavy_features=False):
    print("\n" + "="*80)
    print(f"--- 2. Blackhole Embedding Layer Test (freeze_heavy_features={freeze_heavy_features}) ---".center(80))
    print("="*80)

    # Use the same numeric_feature_dims configuration as in BlackholeConfig
    bh_config = BlackholeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        num_token_id=num_token_id,
        numeric_feature_dims={
            "float64_binary_repr": 64,
            "digit_pos_0": 10,
            "digit_pos_1": 10,
            "log_value": 1,
            "sign": 1,
            "exponent_base10": 1,
            "num_total_digits": 1,
            "num_decimal_places": 1,
            "is_integer_flag": 1,
            "is_positive_flag": 1,
            "is_zero_flag": 1,
            "is_negative_flag": 1,
            "is_power_of_2_flag": 1,
            "format_type_int": 1,
            "format_type_float": 1,
        },
        numeric_embedding_fusion_type="gating", # Can be "add" or "concat"
        numeric_heavy_feature_freeze=freeze_heavy_features,
    )

    embeddings_layer = BlackholeEmbeddings(bh_config)
    embeddings_layer.eval() # Set evaluation mode for testing

    print(f"-> BlackholeEmbeddings initialized with fusion type: '{bh_config.numeric_embedding_fusion_type}'")
    print(f"-> Expected number of numeric features: {bh_config.numeric_input_features}")
    print(f"-> Freezing heavy numeric features: {bh_config.numeric_heavy_feature_freeze}")

    if freeze_heavy_features:
        # Verify freezing
        frozen_params_count = 0
        total_heavy_params = 0
        for name, param in embeddings_layer.named_parameters():
            if 'heavy_numeric_projection' in name or 'float64_binary_repr' in name or 'digit_pos' in name:
                total_heavy_params += 1
                if not param.requires_grad:
                    frozen_params_count += 1
        print(f"Number of frozen parameters (heavy numeric): {frozen_params_count}/{total_heavy_params}")
        if total_heavy_params > 0 and frozen_params_count == total_heavy_params:
            print("-> Verification: 'Heavy' numeric feature layers are correctly frozen!")
        elif total_heavy_params > 0 and frozen_params_count != total_heavy_params:
            print("-> Verification: WARNING! Not all 'heavy' numeric feature parameters were frozen.")
        else:
            print("-> Verification: No 'heavy' numeric feature parameters identified for freezing.")


    sentence = "The value is 123.45 and another is -99."
    encoded_input = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=bh_config.max_position_embeddings,
        return_tensors="pt"
    )

    input_ids = encoded_input['input_ids']
    numeric_values = encoded_input['numeric_values'].double()
    numeric_formats = encoded_input['numeric_formats']

    with torch.no_grad():
        final_embeddings = embeddings_layer(
            input_ids=input_ids,
            numeric_values=numeric_values,
            numeric_formats=numeric_formats
        )

    print(f"Shape of final embeddings from BlackholeEmbeddings: {final_embeddings.shape}")
    print(f"Fragment of first token embedding (first 5 dimensions):\n{final_embeddings[0, 0, :5].tolist()}")

    # Check embedding of [NUM] token
    num_token_positions = (input_ids == num_token_id).nonzero(as_tuple=True)[1]
    if num_token_positions.numel() > 0:
        first_num_pos = num_token_positions[0].item()
        num_embed = final_embeddings[0, first_num_pos, :]
        print(f"\nEmbedding of [NUM] token at position {first_num_pos} (first 5 dimensions):\n{num_embed[:5].tolist()}")
    else:
        print("\nNo [NUM] token in this sentence to check numeric embedding.")

    print("\n--- Additional Embedding Layer Checks ---")
    # Check if embeddings are not all zeros (common issue with bad initialization)
    assert not torch.all(final_embeddings == 0), "Final embeddings are all zeros!"
    print("-> Verification: Embeddings are not all zeros.")

    # Check standard deviation of embeddings to ensure diversity
    # (shouldn't be too close to zero unless it's a padding token or similar)
    embedding_std = torch.std(final_embeddings)
    print(f"Standard deviation of final embeddings: {embedding_std.item():.4f}")
    assert embedding_std > 0.01, "Standard deviation of embeddings is too low, suggesting 'collapsed' embeddings."
    print("-> Verification: Embeddings show reasonable diversity (std > 0.01).")

    # Check magnitude of embeddings (e.g., L2 norm) - should be non-zero
    embedding_norm = torch.norm(final_embeddings)
    print(f"L2 norm of final embeddings: {embedding_norm.item():.4f}")
    assert embedding_norm > 0.1, "L2 norm of embeddings is too low, suggesting very small values."
    print("-> Verification: Embeddings have significant values (norm > 0.1).")

    # Specifically for [NUM] token embedding
    if num_token_positions.numel() > 0:
        num_embed_std = torch.std(num_embed)
        print(f"Standard deviation of [NUM] token embedding: {num_embed_std.item():.4f}")
        assert num_embed_std > 0.01, "Standard deviation of [NUM] token embedding is too low."
        print("-> Verification: [NUM] token embedding shows reasonable diversity.")

    return embeddings_layer, bh_config


# --- Function to test the full Nova model (BlackholeModel) ---
def test_blackhole_model(tokenizer, config):
    print("\n" + "="*80)
    print("--- 3. Testing the full Blackhole (Nova) model ---".center(80))
    print("="*80)

    model = BlackholeModel(config)
    model.eval() # Set evaluation mode

    print(f"-> BlackholeModel initialized. Number of layers: {config.num_hidden_layers}")

    sentence = "The stock market went up by [NUM] percent today."
    encoded_input = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=config.max_position_embeddings,
        return_tensors="pt"
    )

    input_ids = encoded_input['input_ids']
    numeric_values = encoded_input['numeric_values'].double()
    numeric_formats = encoded_input['numeric_formats']
    attention_mask = encoded_input['attention_mask']

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numeric_values=numeric_values,
            numeric_formats=numeric_formats
        )

    sequence_output = outputs.last_hidden_state
    pooled_output = outputs.pooler_output

    print(f"Shape of encoder output (last_hidden_state): {sequence_output.shape}")
    print(f"Shape of pooler output (pooled_output): {pooled_output.shape}")

    # Check that values are not NaN/Inf
    assert not torch.isnan(sequence_output).any(), "Sequence output contains NaN!"
    assert not torch.isinf(sequence_output).any(), "Sequence output contains Inf!"
    if pooled_output is not None:
        assert not torch.isnan(pooled_output).any(), "Pooled output contains NaN!"
        assert not torch.isinf(pooled_output).any(), "Pooled output contains Inf!"
    print("-> Verification: Model outputs are numerically stable (no NaN/Inf).")

    print("\n--- Additional Blackhole Model Output Checks ---")
    # Check that sequence_output is not constant (all same values)
    assert not (sequence_output.max() == sequence_output.min()), "Sequence output is constant (all values are the same)!"
    print("-> Verification: Sequence output is not constant.")

    # Check range of sequence_output values
    seq_output_min = sequence_output.min().item()
    seq_output_max = sequence_output.max().item()
    print(f"Range of sequence output: [{seq_output_min:.4f}, {seq_output_max:.4f}]")
    # We expect values around 0 for a BERT-like model without activation on the last layer
    # Relaxed assertion for untrained model
    assert -20.0 < seq_output_min < 20.0 and -20.0 < seq_output_max < 20.0, \
        f"Sequence output values are outside the expected range [-20.0, 20.0]. Current range: [{seq_output_min:.4f}, {seq_output_max:.4f}]"
    print("-> Verification: Sequence output values are within the expected range.")

    if pooled_output is not None:
        # Check that pooled_output is not constant
        assert not (pooled_output.max() == pooled_output.min()), "Pooler output is constant (all values are the same)!"
        print("-> Verification: Pooler output is not constant.")
        pooled_output_min = pooled_output.min().item()
        pooled_output_max = pooled_output.max().item()
        print(f"Range of pooler output: [{pooled_output_min:.4f}, {pooled_output_max:.4f}]")
        # Relaxed assertion for untrained model
        assert -20.0 < pooled_output_min < 20.0 and -20.0 < pooled_output_max < 20.0, \
            f"Pooler output values are outside the expected range [-20.0, 20.0]. Current range: [{pooled_output_min:.4f}, {pooled_output_max:.4f}]"
        print("-> Verification: Pooler output values are within the expected range.")

    return model

# --- Function to test BlackholeForMaskedLM ---
def test_masked_lm_model(tokenizer, config):
    print("\n" + "="*80)
    print("--- 4. Testing BlackholeForMaskedLM model ---".center(80))
    print("="*80)

    # Use the same configuration, but increase hidden_size and layers for MLM
    mlm_config = BlackholeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4, # You can increase for a more complex test
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        num_token_id=config.num_token_id,
        numeric_feature_dims=config.numeric_feature_dims,
        numeric_embedding_fusion_type=config.numeric_embedding_fusion_type,
    )

    model_mlm = BlackholeForMaskedLM(mlm_config)
    model_mlm.eval() # Set evaluation mode

    print(f"-> BlackholeForMaskedLM initialized.")

    # Original sentence containing a number that your tokenizer will convert to [NUM]
    original_sentence_with_num = "The temperature is 25.5 degrees Celsius."

    encoded_original = tokenizer(
        original_sentence_with_num,
        padding="max_length",
        truncation=True,
        max_length=mlm_config.max_position_embeddings,
        return_tensors="pt"
    )

    # Identify the position of the [NUM] token in the tokenized original sentence
    num_token_id_in_vocab = tokenizer.vocab.get(CUSTOM_SPECIAL_TOKENS["number_token"])
    mask_token_id_in_vocab = tokenizer.vocab.get(CUSTOM_SPECIAL_TOKENS["mask_token"]) # Get mask token ID
    num_token_positions = (encoded_original['input_ids'] == num_token_id_in_vocab).nonzero(as_tuple=True)

    mask_idx = -1
    if num_token_positions[1].numel() > 0:
        mask_idx = num_token_positions[1][0].item() # Take the first [NUM] token position
        print(f"Original sentence: '{original_sentence_with_num}'")
        print(f"Found [NUM] token at position {mask_idx} (ID: {num_token_id_in_vocab}) in original tokenization.")

        # Now, create the masked input for MLM
        # We'll mask the [NUM] token itself for this specific test

        input_ids_masked = encoded_original['input_ids'].clone()
        original_token_id_at_mask_pos = input_ids_masked[0, mask_idx].item()

        # Replace the token at mask_idx with the [MASK] token ID
        input_ids_masked[0, mask_idx] = mask_token_id_in_vocab

        print(f"Masking token at position {mask_idx}. Original ID: {original_token_id_at_mask_pos}, Masked ID: {mask_token_id_in_vocab}")

        # Labels for MLM: -100 for non-masked tokens, original ID for masked tokens
        labels = torch.full(input_ids_masked.shape, -100, dtype=torch.long, device=input_ids_masked.device)
        labels[0, mask_idx] = original_token_id_at_mask_pos # The model should predict the original [NUM] token ID

        # Pass the original numeric_values and numeric_formats.
        # The model will internally align these with the [NUM] token's position.

        print(f"Labels for MLM (fragment): {labels[0, :min(20, labels.shape[1])].tolist()}...")
        print(f"Input IDs for MLM (fragment): {input_ids_masked[0, :min(20, input_ids_masked.shape[1])].tolist()}...")

        with torch.no_grad():
            outputs = model_mlm(
                input_ids=input_ids_masked,
                attention_mask=encoded_original['attention_mask'],
                numeric_values=encoded_original['numeric_values'].double(), # Use original numeric values
                numeric_formats=encoded_original['numeric_formats'], # Use original numeric formats
                labels=labels # Pass labels to calculate loss
            )

        logits = outputs.logits
        loss = outputs.loss

        print(f"Shape of MLM logits: {logits.shape}")
        print(f"Calculated MLM loss: {loss.item()}")
        assert not torch.isnan(logits).any(), "MLM logits contain NaN!"
        assert not torch.isinf(logits).any(), "MLM logits contain Inf!"
        assert not torch.isnan(loss).any(), "MLM loss contain NaN!"
        print("-> Verification: MLM model outputs and loss are numerically stable.")

        print("\n--- Additional MaskedLM Output Checks ---")
        # Check range of logits
        logits_min = logits.min().item()
        logits_max = logits.max().item()
        print(f"Range of MLM logits: [{logits_min:.4f}, {logits_max:.4f}]")
        # Logits usually have a wide range, but extreme values can indicate problems
        # Relaxed assertion for untrained model
        assert -100.0 < logits_min < 100.0 and -100.0 < logits_max < 100.0, \
            f"MLM logits are outside the typical range [-100.0, 100.0]. Current range: [{logits_min:.4f}, {logits_max:.4f}]"
        print("-> Verification: MLM logits are within the expected range.")

        # Check if logits are not constant (means the model is not learning anything meaningful)
        assert not (logits.max() == logits.min()), "MLM logits are constant (all values are the same)!"
        print("-> Verification: MLM logits are not constant.")

        # Check if loss is positive and finite
        assert loss.item() > 0, "MLM loss is not positive!"
        print("-> Verification: MLM loss is positive.")

        # Check entropy of predicted probabilities for the masked token
        # This gives an idea of how "certain" or "random" the untrained predictions are.
        # Higher entropy means more uniform (random) predictions, expected for an untrained model.
        masked_token_logits = logits[0, mask_idx, :]
        masked_token_probs = torch.softmax(masked_token_logits, dim=-1)
        # Add epsilon for log(0)
        entropy = -torch.sum(masked_token_probs * torch.log(masked_token_probs + 1e-9)).item()
        print(f"Entropy of masked token prediction: {entropy:.4f}")
        # For an untrained model, entropy should be relatively high
        # Relaxed assertion for untrained model's initial state
        assert entropy > 0.001, "Entropy of masked token prediction is too low, suggesting premature convergence or a problem."
        print("-> Verification: Entropy of masked token prediction is reasonable for an untrained model.")

    else:
        print("No [NUM] token in the example sentence to mask for MLM test.")

    return model_mlm

# --- Function to test BlackholeForSequenceClassification ---
def test_sequence_classification_model(tokenizer, config):
    print("\n" + "="*80)
    print("--- 5. Testing BlackholeForSequenceClassification model ---".center(80))
    print("="*80)

    num_labels = 3 # Example number of classes (e.g., low, medium, high)
    classification_config = BlackholeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        num_token_id=config.num_token_id,
        numeric_feature_dims=config.numeric_feature_dims,
        numeric_embedding_fusion_type=config.numeric_embedding_fusion_type,
        num_labels=num_labels,
        problem_type="single_label_classification",
        classifier_dropout=0.1, # Can be set to another float value, e.g., 0.1, 0.0, or pass config.hidden_dropout_prob
    )

    model_clf = BlackholeForSequenceClassification(classification_config)
    model_clf.eval() # Set evaluation mode

    print(f"-> BlackholeForSequenceClassification initialized. Number of labels: {num_labels}")

    sentence = "The financial report showed a profit of [NUM] million."
    encoded_input = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=classification_config.max_position_embeddings,
        return_tensors="pt"
    )

    labels = torch.tensor([1], dtype=torch.long) # Example label for class 1

    with torch.no_grad():
        outputs = model_clf(
            input_ids=encoded_input['input_ids'],
            attention_mask=encoded_input['attention_mask'],
            numeric_values=encoded_input['numeric_values'].double(),
            numeric_formats=encoded_input['numeric_formats'],
            labels=labels
        )

    logits = outputs.logits
    loss = outputs.loss

    print(f"Shape of classification logits: {logits.shape}")
    print(f"Calculated classification loss: {loss.item()}")
    assert not torch.isnan(logits).any(), "Classification logits contain NaN!"
    assert not torch.isinf(logits).any(), "Classification logits contain Inf!"
    assert not torch.isnan(loss).any(), "Classification loss contain NaN!"
    print("-> Verification: Classification model outputs and loss are numerically stable.")

    print("\n--- Additional Sequence Classification Output Checks ---")
    # Check range of logits
    logits_min = logits.min().item()
    logits_max = logits.max().item()
    print(f"Range of classification logits: [{logits_min:.4f}, {logits_max:.4f}]")
    # For classification logits, the expected range is usually around -5 to 5 for an untrained model
    assert -10.0 < logits_min < 10.0 and -10.0 < logits_max < 10.0, \
        "Classification logits are outside the typical range [-10.0, 10.0]."
    print("-> Verification: Classification logits are within the expected range.")

    # Check that logits are not constant
    assert not (logits.max() == logits.min()), "Classification logits are constant (all values are the same)!"
    print("-> Verification: Classification logits are not constant.")

    # Check if loss is positive and finite
    assert loss.item() > 0, "Classification loss is not positive!"
    print("-> Verification: Classification loss is positive.")

    # Check that probabilities sum to 1 (after softmax)
    probabilities = torch.softmax(logits, dim=-1)
    prob_sum = probabilities.sum().item()
    print(f"Suma prawdopodobieństw klasyfikacji: {prob_sum:.4f}")
    assert torch.isclose(torch.tensor(prob_sum), torch.tensor(1.0), atol=1e-6), \
        "Classification probabilities do not sum to 1 after softmax!"
    print("-> Verification: Classification probabilities sum to 1.")

    # Check that probabilities are within the range [0, 1]
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1), \
        "Classification probabilities are not within the range [0, 1]!"
    print("-> Verification: Classification probabilities are within the range [0, 1].")

    return model_clf


# --- Main program execution ---
if __name__ == "__main__":
    tokenizer_output_dir = "./blackhole_tokenizer_demo"

    # 1. Configure and test the tokenizer (train/load)
    loaded_tokenizer, num_token_id, _ = setup_tokenizer(tokenizer_output_dir)

    # 2. Configure the base model configuration
    # Using smaller values for faster tests
    base_config = BlackholeConfig(
        vocab_size=loaded_tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        pad_token_id=loaded_tokenizer.pad_token_id,
        num_token_id=num_token_id,
        numeric_feature_dims={
            "float64_binary_repr": 64,
            "digit_pos_0": 10,
            "digit_pos_1": 10,
            "log_value": 1,
            "sign": 1,
            "exponent_base10": 1,
            "num_total_digits": 1,
            "num_decimal_places": 1,
            "is_integer_flag": 1,
            "is_positive_flag": 1,
            "is_zero_flag": 1,
            "is_negative_flag": 1,
            "is_power_of_2_flag": 1,
            "format_type_int": 1,
            "format_type_float": 1,
        },
        numeric_embedding_fusion_type="gating",
    )

    # 3. Test the embedding layer with and without freezing heavy features
    # This will run the test twice, showing both scenarios
    test_embedding_layer(loaded_tokenizer, num_token_id, freeze_heavy_features=False)
    test_embedding_layer(loaded_tokenizer, num_token_id, freeze_heavy_features=True)

    # 4. Test the base Blackhole model (Encoder)
    _ = test_blackhole_model(loaded_tokenizer, base_config)

    # 5. Test the Masked Language Modeling model
    _ = test_masked_lm_model(loaded_tokenizer, base_config)

    # 6. Test the Sequence Classification model
    _ = test_sequence_classification_model(loaded_tokenizer, base_config)

    # Optional: Clean up the tokenizer directory
    if os.path.exists(tokenizer_output_dir):
        try:
            # shutil.rmtree(tokenizer_output_dir) # Uncomment if you want to delete files after tests
            print(f"\nCleaned up tokenizer directory: {tokenizer_output_dir}")
        except OSError as e:
            print(f"\nError deleting directory {tokenizer_output_dir}: {e}")

    print("\n" + "="*80)
    print("--- All Nova architecture tests completed successfully! ---".center(80))
    print("="*80)

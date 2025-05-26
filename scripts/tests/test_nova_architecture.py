# test_nova_architecture.py
import os
import sys
import torch
import numpy as np
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

from blackhole.nova_hugging_face import *
from blackhole.nova_hugging_face.configuration_nova import BlackholeConfig

from blackhole.embadding_hugging_face import BlackholeEmbeddings, BlackholeConfig
from blackhole.tokenizer_hugging_face import BlackholeTokenizer, CUSTOM_SPECIAL_TOKENS

# --- Funkcja pomocnicza do ustawiania tokenizatora (jak w Twoim skrypcie) ---
def setup_tokenizer(output_dir="./blackhole_tokenizer_demo"):
    """Inicjalizuje i trenuje BlackholeTokenizer, a następnie go zapisuje i ładuje."""
    print("\n" + "="*80)
    print("--- 1. Konfiguracja i Inicjalizacja Tokenizera Blackhole ---".center(80))
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
        "The value is 0xAF and 0b101.", # Przykładowe liczby szesnastkowe/binarne, jeśli tokenizer je wspiera
        "It's about -3.14 degrees."
    ]
    print(f"Trenowanie tokenizera na {len(sample_texts_for_training)} przykładowych zdaniach...")
    tokenizer.train_tokenizer(sample_texts_for_training, vocab_size=8000, min_freq=1)

    num_token_id = tokenizer.vocab.get(CUSTOM_SPECIAL_TOKENS["number_token"])
    if num_token_id is None:
        raise ValueError(f"Token '{CUSTOM_SPECIAL_TOKENS['number_token']}' nie znaleziono w słowniku. Upewnij się, że został dodany podczas trenowania.")
    print(f"-> ID dla specjalnego tokena {CUSTOM_SPECIAL_TOKENS['number_token']}: {num_token_id}")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    loaded_tokenizer = BlackholeTokenizer.from_pretrained(output_dir)
    print(f"-> Tokenizator załadowany pomyślnie z '{output_dir}'")
    return loaded_tokenizer, num_token_id, output_dir

# --- Funkcja do testowania embeddingów ---
def test_embedding_layer(tokenizer, num_token_id, freeze_heavy_features=False):
    print("\n" + "="*80)
    print("--- 2. Testowanie Warstwy Osadzania Blackhole ---".center(80))
    print("="*80)

    # Używamy tej samej konfiguracji numeric_feature_dims, co w BlackholeConfig
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
        numeric_embedding_fusion_type="gating", # Możesz zmienić na "add" lub "concat"
        numeric_heavy_feature_freeze=freeze_heavy_features,
    )

    embeddings_layer = BlackholeEmbeddings(bh_config)
    embeddings_layer.eval() # Ustaw tryb ewaluacji do testów

    print(f"-> BlackholeEmbeddings zainicjalizowane z typem fuzji: '{bh_config.numeric_embedding_fusion_type}'")
    print(f"-> Oczekiwana liczba cech numerycznych: {bh_config.numeric_input_features}")
    print(f"-> Zamrażanie ciężkich cech numerycznych: {bh_config.numeric_heavy_feature_freeze}")

    if freeze_heavy_features:
        # Weryfikacja zamrażania
        frozen_params_count = 0
        total_heavy_params = 0
        for name, param in embeddings_layer.named_parameters():
            if 'heavy_numeric_projection' in name or 'float64_binary_repr' in name or 'digit_pos' in name:
                total_heavy_params += 1
                if not param.requires_grad:
                    frozen_params_count += 1
        print(f"Liczba zamrożonych parametrów (heavy numeric): {frozen_params_count}/{total_heavy_params}")
        if total_heavy_params > 0 and frozen_params_count == total_heavy_params:
            print("-> Weryfikacja: Warstwy 'heavy' cech numerycznych są prawidłowo zamrożone!")
        elif total_heavy_params > 0 and frozen_params_count != total_heavy_params:
            print("-> Weryfikacja: UWAGA! Nie wszystkie parametry 'heavy' cech numerycznych zostały zamrożone.")
        else:
            print("-> Weryfikacja: Brak zidentyfikowanych parametrów 'heavy' cech numerycznych do zamrożenia.")


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

    print(f"Kształt końcowych osadzeń z BlackholeEmbeddings: {final_embeddings.shape}")
    print(f"Fragment osadzenia pierwszego tokena (pierwsze 5 wymiarów):\n{final_embeddings[0, 0, :5].tolist()}")

    # Sprawdzenie osadzenia tokenu [NUM]
    num_token_positions = (input_ids == num_token_id).nonzero(as_tuple=True)[1]
    if num_token_positions.numel() > 0:
        first_num_pos = num_token_positions[0].item()
        num_embed = final_embeddings[0, first_num_pos, :]
        print(f"\nOsadzenie tokenu [NUM] na pozycji {first_num_pos} (pierwsze 5 wymiarów):\n{num_embed[:5].tolist()}")
    else:
        print("\nBrak tokena [NUM] w tym zdaniu do sprawdzenia osadzenia numerycznego.")

    return embeddings_layer, bh_config


# --- Funkcja do testowania pełnego modelu Nova (BlackholeModel) ---
def test_blackhole_model(tokenizer, config):
    print("\n" + "="*80)
    print("--- 3. Testowanie pełnego modelu Blackhole (Nova) ---".center(80))
    print("="*80)

    model = BlackholeModel(config)
    model.eval() # Ustaw tryb ewaluacji

    print(f"-> BlackholeModel zainicjalizowany. Liczba warstw: {config.num_hidden_layers}")

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

    print(f"Kształt wyjścia encoder'a (last_hidden_state): {sequence_output.shape}")
    print(f"Kształt wyjścia pooler'a (pooled_output): {pooled_output.shape}")

    # Sprawdzenie, czy wartości nie są NaN/Inf
    assert not torch.isnan(sequence_output).any(), "Sequence output zawiera NaN!"
    assert not torch.isinf(sequence_output).any(), "Sequence output zawiera Inf!"
    if pooled_output is not None:
        assert not torch.isnan(pooled_output).any(), "Pooled output zawiera NaN!"
        assert not torch.isinf(pooled_output).any(), "Pooled output zawiera Inf!"
    print("-> Weryfikacja: Wyjścia modelu są numerycznie stabilne (brak NaN/Inf).")

    return model

# --- Funkcja do testowania BlackholeForMaskedLM ---
def test_masked_lm_model(tokenizer, config):
    print("\n" + "="*80)
    print("--- 4. Testowanie modelu BlackholeForMaskedLM ---".center(80))
    print("="*80)

    # Używamy tej samej konfiguracji, ale zwiększamy hidden_size i warstwy dla MLM
    mlm_config = BlackholeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4, # Możesz zwiększyć dla bardziej złożonego testu
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        num_token_id=config.num_token_id,
        numeric_feature_dims=config.numeric_feature_dims,
        numeric_embedding_fusion_type=config.numeric_embedding_fusion_type,
    )

    model_mlm = BlackholeForMaskedLM(mlm_config)
    model_mlm.eval() # Ustaw tryb ewaluacji

    print(f"-> BlackholeForMaskedLM zainicjalizowany.")

    # Przykładowe zdanie z zamaskowanym tokenem (np. [NUM])
    # Pamiętaj, że do treningu MLM potrzebujesz etykiet (-100 dla niezamaskowanych, ID tokena dla zamaskowanych)
    original_sentence = "The temperature is 25.5 degrees Celsius."
    tokens = tokenizer.tokenize(original_sentence)

    # Znajdź pozycję liczby i zamaskuj ją
    mask_idx = -1
    for i, token in enumerate(tokens):
        if tokenizer._is_number(token): # Używamy wewnętrznej metody do identyfikacji liczby
            mask_idx = i
            break
    
    if mask_idx != -1:
        masked_sentence = original_sentence.replace("25.5", CUSTOM_SPECIAL_TOKENS["mask_token"], 1)
        print(f"Oryginalne zdanie: '{original_sentence}'")
        print(f"Zamaskowane zdanie dla MLM: '{masked_sentence}'")

        encoded_input = tokenizer(
            masked_sentence,
            padding="max_length",
            truncation=True,
            max_length=mlm_config.max_position_embeddings,
            return_tensors="pt"
        )
        
        # Tworzenie etykiet dla MLM
        labels = encoded_input['input_ids'].clone()
        # Wartość -100 oznacza ignorowanie tokena w obliczeniach loss
        labels[labels != tokenizer.mask_token_id] = -100
        
        # Jeśli zamaskowaliśmy [NUM], chcemy, żeby model przewidział jego ID
        # W prostym teście, to jest trudne bez prawdziwego treningu,
        # ale labels dla MLM powinny mieć rzeczywiste ID tokena do przewidzenia.
        # W tym przykładzie, po prostu sprawdzimy kształt wyjścia.
        
        # Możemy sztucznie ustawić etykietę dla testu (tylko do sprawdzenia kształtu loss)
        if mask_idx != -1:
            original_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(original_sentence))
            labels[0, mask_idx] = original_tokens_ids[mask_idx]
            
        print(f"Etykiety dla MLM (fragment): {labels[0, :min(20, labels.shape[1])].tolist()}...")
        print(f"Input IDs dla MLM (fragment): {encoded_input['input_ids'][0, :min(20, encoded_input['input_ids'].shape[1])].tolist()}...")


        with torch.no_grad():
            outputs = model_mlm(
                input_ids=encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                numeric_values=encoded_input['numeric_values'].double(),
                numeric_formats=encoded_input['numeric_formats'],
                labels=labels # Przekazujemy etykiety do obliczenia loss
            )

        logits = outputs.logits
        loss = outputs.loss

        print(f"Kształt logits MLM: {logits.shape}")
        print(f"Obliczony loss MLM: {loss.item()}")
        assert not torch.isnan(logits).any(), "MLM logits zawiera NaN!"
        assert not torch.isinf(logits).any(), "MLM logits zawiera Inf!"
        assert not torch.isnan(loss).any(), "MLM loss zawiera NaN!"
        print("-> Weryfikacja: Wyjścia modelu MLM i loss są numerycznie stabilne.")

    else:
        print("Brak liczby w przykładowym zdaniu do zamaskowania.")

    return model_mlm

# --- Funkcja do testowania BlackholeForSequenceClassification ---
def test_sequence_classification_model(tokenizer, config):
    print("\n" + "="*80)
    print("--- 5. Testowanie modelu BlackholeForSequenceClassification ---".center(80))
    print("="*80)

    num_labels = 3 # Przykładowa liczba klas (np. niski, średni, wysoki)
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
        num_labels=num_labels, # Dodaj liczbę etykiet do konfiguracji
        problem_type="single_label_classification" # lub "regression", "multi_label_classification"
    )

    model_clf = BlackholeForSequenceClassification(classification_config)
    model_clf.eval() # Ustaw tryb ewaluacji

    print(f"-> BlackholeForSequenceClassification zainicjalizowany. Liczba etykiet: {num_labels}")

    sentence = "The financial report showed a profit of [NUM] million."
    encoded_input = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=classification_config.max_position_embeddings,
        return_tensors="pt"
    )

    labels = torch.tensor([1], dtype=torch.long) # Przykładowa etykieta dla klasy 1

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

    print(f"Kształt logits klasyfikacji: {logits.shape}")
    print(f"Obliczony loss klasyfikacji: {loss.item()}")
    assert not torch.isnan(logits).any(), "Classification logits zawiera NaN!"
    assert not torch.isinf(logits).any(), "Classification logits zawiera Inf!"
    assert not torch.isnan(loss).any(), "Classification loss zawiera NaN!"
    print("-> Weryfikacja: Wyjścia modelu klasyfikacji i loss są numerycznie stabilne.")

    return model_clf


# --- Główna część programu ---
if __name__ == "__main__":
    tokenizer_output_dir = "./blackhole_tokenizer_demo"

    # 1. Skonfiguruj i przetestuj tokenizator (ponowne trenowanie/ładowanie)
    loaded_tokenizer, num_token_id, _ = setup_tokenizer(tokenizer_output_dir)

    # 2. Skonfiguruj podstawową konfigurację dla modelu
    # Używamy mniejszych wartości dla szybszych testów
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

    # 3. Testuj warstwę embeddingów z i bez zamrażania
    _, _ = test_embedding_layer(loaded_tokenizer, num_token_id, freeze_heavy_features=False)
    _, _ = test_embedding_layer(loaded_tokenizer, num_token_id, freeze_heavy_features=True)

    # 4. Testuj bazowy model Blackhole (Encoder)
    _ = test_blackhole_model(loaded_tokenizer, base_config)

    # 5. Testuj model do Masked Language Modeling
    _ = test_masked_lm_model(loaded_tokenizer, base_config)

    # 6. Testuj model do Klasyfikacji Sekwencji
    _ = test_sequence_classification_model(loaded_tokenizer, base_config)

    # Opcjonalnie: Posprzątaj katalog tokenizatora
    if os.path.exists(tokenizer_output_dir):
        try:
            # shutil.rmtree(tokenizer_output_dir) # Odkomentuj, jeśli chcesz usuwać pliki po testach
            print(f"\nPosprzątano katalog tokenizatora: {tokenizer_output_dir}")
        except OSError as e:
            print(f"\nBłąd podczas usuwania katalogu {tokenizer_output_dir}: {e}")

    print("\n" + "="*80)
    print("--- Wszystkie testy architektury Nova zakończone pomyślnie! ---".center(80))
    print("="*80)
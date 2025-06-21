import os, sys
import torch
import numpy as np
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..\..')))

# Importujemy zmienione BlackholeEmbeddings i BlackholeConfig
from blackhole.embadding_hugging_face import BlackholeEmbeddings, BlackholeConfig
from blackhole.tokenizer_hugging_face import BlackholeTokenizer, CUSTOM_SPECIAL_TOKENS

# --- Konfiguracja i Inicjalizacja Tokenizera Blackhole ---
def setup_tokenizer():
    """Inicjalizuje i trenuje BlackholeTokenizer."""
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
        "This sentence has no numbers."
    ]
    print(f"Trenowanie tokenizera na {len(sample_texts_for_training)} przykładowych zdaniach...")
    tokenizer.train_tokenizer(sample_texts_for_training, vocab_size=8000, min_freq=1)

    num_token_id = tokenizer.vocab.get(CUSTOM_SPECIAL_TOKENS["number_token"])
    if num_token_id is None:
        raise ValueError(f"Token '{CUSTOM_SPECIAL_TOKENS['number_token']}' nie znaleziono w słowniku. Upewnij się, że został dodany podczas trenowania.")
    print(f"-> ID dla specjalnego tokena {CUSTOM_SPECIAL_TOKENS['number_token']}: {num_token_id}")

    output_dir = "./blackhole_tokenizer_demo"
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    loaded_tokenizer = BlackholeTokenizer.from_pretrained(output_dir)
    print(f"-> Tokenizator załadowany pomyślnie z '{output_dir}'")
    return loaded_tokenizer, num_token_id, output_dir

# --- Konfiguracja i Inicjalizacja Warstwy Osadzania Blackhole ---
def setup_embeddings_layer(tokenizer, num_token_id):
    """Konfiguruje i inicjalizuje warstwę BlackholeEmbeddings z 96 cechami numerycznymi."""
    print("\n" + "="*80)
    print("--- 2. Konfiguracja i Inicjalizacja Warstwy Osadzania Blackhole ---".center(80))
    print("="*80)

    # Używamy DOKŁADNIE tej samej konfiguracji numeric_feature_dims, co w hugging_embedding.py
    # aby suma cech wynosiła 96.
    bh_config = BlackholeConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,             # Przykładowy wymiar osadzania (zmniejszony dla testu)
        max_position_embeddings=128, # Maksymalna długość sekwencji
        pad_token_id=tokenizer.pad_token_id,
        num_token_id=num_token_id,   # ID tokena [NUM]
        numeric_feature_dims={
            # HEAVY LAYERS / Highly Informative Features (64 + 20 = 84 cechy)
            "float64_binary_repr": 64,
            "digit_pos_0": 10,
            "digit_pos_1": 10,

            # LIGHT LAYERS / Simpler Informative Features (5 + 7 = 12 cech)
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
        # numeric_projection_intermediate_size_ratio=0.5, # Domyślna wartość w config
    )

    embeddings_layer = BlackholeEmbeddings(bh_config)
    print(f"-> BlackholeEmbeddings zainicjalizowane z typem fuzji: '{bh_config.numeric_embedding_fusion_type}'")
    print(f"-> Oczekiwana liczba cech numerycznych: {bh_config.numeric_input_features}")
    return embeddings_layer, bh_config

# --- Przetwarzanie i Osadzanie Przykładowego Zdania ---
def process_and_embed_sentence(sentence, tokenizer, embeddings_layer, bh_config, num_token_id):
    """Tokenizuje, osadza i de-tokenizuje przykładowe zdanie."""
    print("\n" + "="*80)
    print("--- 3. Przetwarzanie i Osadzanie Przykładowego Zdania ---".center(80))
    print("="*80)
    print(f"Oryginalne zdanie: '{sentence}'")

    # Tokenizacja zdania
    # Ważne: `return_tensors="pt"` zapewnia, że dostaniemy tensory PyTorch
    encoded_input = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=bh_config.max_position_embeddings,
        return_tensors="pt"
    )

    input_ids = encoded_input['input_ids']
    # Upewnij się, że numeric_values są typu float64, jeśli tokenizator nie zwraca ich jako takich
    # Tokenizator powinien to robić automatycznie, ale to jest bezpieczne sprawdzenie
    numeric_values = encoded_input['numeric_values'].double()
    numeric_formats = encoded_input['numeric_formats']
    attention_mask = encoded_input['attention_mask']

    print(f"\n--- Wyjście Tokenizatora ---")
    print(f"Input IDs (tokeny):\n{input_ids[0].tolist()}")
    print(f"Wartości liczbowe:\n{numeric_values[0].tolist()}")
    print(f"Formaty liczbowe:\n{numeric_formats[0].tolist()} (0=int, 1=float, 2=naukowe, -1=padding)")
    print(f"Maska uwagi:\n{attention_mask[0].tolist()}")

    # Konwertuj ID z powrotem na tokeny dla czytelności
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    print(f"Tokeny tekstowe:\n{tokens}")

    # Pokaż wartości liczbowe przypisane do tokenów [NUM]
    num_token_positions = (input_ids == num_token_id).nonzero(as_tuple=True)[1]
    if num_token_positions.numel() > 0:
        print("\n--- Szczegóły tokenów numerycznych ---")
        for pos in num_token_positions:
            val = numeric_values[0, pos.item()].item()
            fmt_id = numeric_formats[0, pos.item()].item()
            fmt_str = {0: "integer", 1: "float", 2: "scientific", 3: "hexadecimal"}.get(fmt_id, "unknown/padding")
            print(f"  - Na pozycji {pos.item()} ({tokens[pos.item()]}): Wartość={val}, Format={fmt_str} (ID: {fmt_id})")
    else:
        print("-> Brak tokena [NUM] w tym zdaniu.")

    # Przekaż przetworzone dane do warstwy osadzania
    print(f"\n--- Wyjście BlackholeEmbeddings (Osadzenie) ---")
    final_embeddings = embeddings_layer(
        input_ids=input_ids,
        numeric_values=numeric_values,
        numeric_formats=numeric_formats
    )

    print(f"Kształt końcowych osadzeń: {final_embeddings.shape}")
    print(f"Fragment osadzenia pierwszego tokena (pierwsze 5 wymiarów):\n{final_embeddings[0, 0, :5].tolist()}")

    # Dodatkowa weryfikacja: Sprawdź osadzenie tokenu [NUM]
    if num_token_positions.numel() > 0:
        num_embed_pos = num_token_positions[0].item() # Bierzemy pierwszą pozycję [NUM]
        num_embed = final_embeddings[0, num_embed_pos, :]
        print(f"\nOsadzenie tokenu [NUM] na pozycji {num_embed_pos} (pierwsze 5 wymiarów):\n{num_embed[:5].tolist()}")
        # Możesz dodać więcej asercji, np. czy to osadzenie jest różne od osadzeń tekstowych.

    # --- Detokenizacja ---
    print(f"\n" + "="*80)
    print("--- Detokenizacja ---".center(80))
    print("="*80)
    decoded_text_with_special = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    decoded_text_clean = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    print(f"Odtworzony tekst (z tokenami specjalnymi):  '{decoded_text_with_special}'")
    print(f"Odtworzony tekst (bez tokenów specjalnych): '{decoded_text_clean}'")

    # Weryfikacja (po usunięciu spacji z początku/końca)
    if sentence.strip() == decoded_text_clean.strip():
        print("-> Weryfikacja: Oryginalne zdanie i czysty odtworzony tekst są identyczne!")
    else:
        print("-> Weryfikacja: ROZBIEŻNOŚĆ między oryginalnym zdaniem a czystym odtworzonym tekstem!")
        print(f"   Oryginalne: '{sentence.strip()}'")
        print(f"   Odtworzone: '{decoded_text_clean.strip()}'")


# --- Główna część programu ---
if __name__ == "__main__":
    # 1. Skonfiguruj tokenizator
    loaded_tokenizer, num_token_id, tokenizer_output_dir = setup_tokenizer()

    # 2. Skonfiguruj warstwę osadzania
    embeddings_layer, bh_config = setup_embeddings_layer(loaded_tokenizer, num_token_id)

    # 3. Przetwórz przykładowe zdanie
    example_sentence_with_number = "The temperature is 30.2 degrees Celsius."
    example_sentence_no_number = "This sentence has no numbers."
    example_sentence_multiple_numbers = "I have 12 apples and 5.5 oranges. That's 17.5 total."
    example_sentence_scientific = "The speed of light is 2.99792458e8 m/s."
    example_sentence_negative = "The stock dropped -1.5%."

    # Test z liczbą
    process_and_embed_sentence(example_sentence_with_number, loaded_tokenizer, embeddings_layer, bh_config, num_token_id)
    # Test bez liczby
    process_and_embed_sentence(example_sentence_no_number, loaded_tokenizer, embeddings_layer, bh_config, num_token_id)
    # Test z wieloma liczbami
    process_and_embed_sentence(example_sentence_multiple_numbers, loaded_tokenizer, embeddings_layer, bh_config, num_token_id)
    # Test z notacją naukową
    process_and_embed_sentence(example_sentence_scientific, loaded_tokenizer, embeddings_layer, bh_config, num_token_id)
    # Test z liczbą ujemną
    process_and_embed_sentence(example_sentence_negative, loaded_tokenizer, embeddings_layer, bh_config, num_token_id)


    # Opcjonalnie: Posprzątaj katalog tokenizatora
    if os.path.exists(tokenizer_output_dir):
        try:
            shutil.rmtree(tokenizer_output_dir)
            print(f"\nPosprzątano katalog tokenizatora: {tokenizer_output_dir}")
        except OSError as e:
            print(f"\nBłąd podczas usuwania katalogu {tokenizer_output_dir}: {e}")
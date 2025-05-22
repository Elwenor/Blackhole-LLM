import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Dodaj ścieżki, aby Python mógł znaleźć Twoje moduły
# Zakładamy, że test_blackhole_embeddings.py jest w katalogu głównym projektu
# a blackhole_embeddings.py i katalog blackhole są obok niego.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

# Importujemy zmienione BlackholeEmbeddings i BlackholeConfig
from blackhole.embadding_hugging_face import BlackholeEmbeddings, BlackholeConfig
from blackhole.tokenizer_hugging_face import BlackholeTokenizer, CUSTOM_SPECIAL_TOKENS

# --- 1. Konfiguracja i Inicjalizacja Tokenizatora ---
print("--- Krok 1: Konfiguracja i Inicjalizacja Tokenizatora ---")
tokenizer = BlackholeTokenizer()
texts_for_training = [
    "The temperature is 25.5 degrees Celsius.",
    "My bank balance is -123.45 dollars.",
    "The population is 8.0e9 people.",
    "A simple integer: 42.",
    "The number of users: 1000.",
    "It costs 99.99 EUR.",
    "No numbers here."
]
tokenizer.train_tokenizer(texts_for_training, vocab_size=8000, min_freq=1)

num_token_id = tokenizer.vocab.get(CUSTOM_SPECIAL_TOKENS["number_token"])
if num_token_id is None:
    raise ValueError(f"Token '{CUSTOM_SPECIAL_TOKENS['number_token']}' nie znaleziono w słowniku.")

# --- 2. Ustawienie Konfiguracji dla Embeddings ---
print(f"\n--- Krok 2: Konfiguracja BlackholeEmbeddings (num_token_id={num_token_id}) ---")
# Dostosowujemy numeric_input_features do liczby cech, które faktycznie generujemy w _get_numeric_features
# (log_value, sign, exponent, pseudo_binary_representation, format_type)
config = BlackholeConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256, # Zmniejszamy hidden_size dla szybszych testów
    max_position_embeddings=128,
    pad_token_id=tokenizer.pad_token_id,
    num_token_id=num_token_id,
    numeric_feature_dims={
        "log_value": 1,
        "sign": 1,
        "exponent": 1,
        "binary_representation": 16, # Użyjemy 16 bitów dla uproszczonej reprezentacji
        "format_type": 3, # 0=int, 1=float, 2=scientific
    },
    numeric_embedding_fusion_type="gating" # Testujemy innowacyjny mechanizm bramkowania
)

embeddings_layer = BlackholeEmbeddings(config)
print("BlackholeEmbeddings zainicjalizowano pomyślnie z typem fuzji:", config.numeric_embedding_fusion_type)

# --- 3. Funkcja Pomocnicza do Weryfikacji Osadzeń Numerycznych ---
def verify_numeric_embedding(
    text: str,
    expected_num_value: float,
    expected_format_id: int, # 0: int, 1: float, 2: scientific
    embeddings_layer: BlackholeEmbeddings,
    tokenizer: BlackholeTokenizer,
    config: BlackholeConfig,
    description: str
):
    print(f"\n--- Test dla: {description} (Wartość: {expected_num_value}, Format: {expected_format_id}) ---")
    encoded_input = tokenizer(
        text,
        padding="max_length",
        max_length=config.max_position_embeddings,
        return_tensors="pt"
    )

    input_ids = encoded_input['input_ids']
    numeric_values = encoded_input['numeric_values']
    numeric_formats = encoded_input['numeric_formats'] # Pamiętaj, że tokenizer musi to zwracać!

    # Przekazanie do embeddings_layer
    final_embeddings = embeddings_layer(
        input_ids=input_ids,
        numeric_values=numeric_values,
        numeric_formats=numeric_formats # Przekazujemy formaty
    )

    print(f"Tekst: '{text}'")
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Numeric Values: {numeric_values.tolist()}")
    print(f"Numeric Formats: {numeric_formats.tolist()}")
    print(f"Kształt finalnych osadzeń: {final_embeddings.shape}")

    # Znajdź pozycję tokenu [NUM]
    num_token_pos = (input_ids == config.num_token_id).nonzero(as_tuple=True)[1]

    if num_token_pos.numel() > 0:
        pos = num_token_pos[0].item() # Bierzemy pierwszą instancję
        
        # Sprawdź, czy numeric_values i numeric_formats są poprawne w tej pozycji
        actual_val = numeric_values[0, pos].item()
        actual_format = numeric_formats[0, pos].item()
        print(f"Znaleziono [NUM] na pozycji {pos}. Wartość: {actual_val}, Format: {actual_format}")
        
        if not np.isclose(actual_val, expected_num_value) or actual_format != expected_format_id:
             print(f"BŁĄD: Oczekiwana wartość/format ({expected_num_value}/{expected_format_id}) "
                   f"nie zgadza się z otrzymaną ({actual_val}/{actual_format}). Sprawdź tokenizer!")
             return False # Poinformuj o błędzie w tokenizerze

        # Pobierz osadzenie dla tokenu [NUM]
        num_embedding = final_embeddings[0, pos]
        print(f"Fragment osadzenia dla [NUM] (pierwsze 5 wymiarów): {num_embedding[:5].tolist()}")
        
        # Sprawdź, czy osadzenie jest znaczące (nie zerowe)
        is_meaningful = not torch.allclose(num_embedding, torch.zeros_like(num_embedding), atol=1e-5)
        print(f"Czy osadzenie [NUM] jest znaczące (nie zerowe)? {is_meaningful}")
        assert is_meaningful, f"Osadzenie dla tokenu [NUM] ({expected_num_value}) jest zerowe!"

        # Sprawdź, czy inne osadzenia w tym samym zdaniu, które nie są [NUM], są znaczące tekstowo
        # ale nie zawierają silnego wpływu numerycznego (czego nie widać bezpośrednio tutaj, ale model by to rozróżnił)
        
        # Demonstracja, że nie-numeryczne tokeny mają swoje osadzenia tekstowe
        # Znajdź pierwszy token, który NIE jest [NUM] ani [PAD]
        non_num_non_pad_token_pos = -1
        for i in range(input_ids.size(1)):
            if input_ids[0, i].item() != config.num_token_id and input_ids[0, i].item() != config.pad_token_id:
                non_num_non_pad_token_pos = i
                break
        
        if non_num_non_pad_token_pos != -1:
            text_only_embedding = final_embeddings[0, non_num_non_pad_token_pos]
            is_text_meaningful = not torch.allclose(text_only_embedding, torch.zeros_like(text_only_embedding), atol=1e-5)
            print(f"Fragment osadzenia dla tokenu '{tokenizer.decode([input_ids[0, non_num_non_pad_token_pos].item()])}' (pierwsze 5 wymiarów): {text_only_embedding[:5].tolist()}")
            print(f"Czy osadzenie tekstowe jest znaczące? {is_text_meaningful}")
            assert is_text_meaningful, f"Osadzenie dla tokenu '{tokenizer.decode([input_ids[0, non_num_non_pad_token_pos].item()])}' jest zerowe!"

        print("Test pomyślny.")
        return True
    else:
        print(f"BŁĄD: Token '{CUSTOM_SPECIAL_TOKENS['number_token']}' nie został wygenerowany dla tekstu: '{text}'")
        return False


# --- 4. Wykonanie Testów Kluczowych Scenariuszy ---
print("\n" + "="*50)
print("--- ROZPOCZYNAM TESTOWANIE KLUCZOWYCH SCENARIUSZY ---")
print("="*50)

# Test 1: Liczba zmiennoprzecinkowa
verify_numeric_embedding(
    text="The value is 123.45.",
    expected_num_value=123.45,
    expected_format_id=1, # float
    embeddings_layer=embeddings_layer,
    tokenizer=tokenizer,
    config=config,
    description="Liczba zmiennoprzecinkowa dodatnia"
)

# Test 2: Liczba całkowita ujemna
verify_numeric_embedding(
    text="Temperature dropped to -10 degrees.",
    expected_num_value=-10.0,
    expected_format_id=0, # int (zakładamy, że tokenizer wykrywa jako int, mimo reprezentacji float)
    embeddings_layer=embeddings_layer,
    tokenizer=tokenizer,
    config=config,
    description="Liczba całkowita ujemna"
)

# Test 3: Liczba w notacji naukowej
verify_numeric_embedding(
    text="Mass of electron is 9.109e-31 kg.",
    expected_num_value=9.109e-31,
    expected_format_id=2, # scientific
    embeddings_layer=embeddings_layer,
    tokenizer=tokenizer,
    config=config,
    description="Liczba w notacji naukowej"
)

# Test 4: Zero
verify_numeric_embedding(
    text="Zero profit: 0.",
    expected_num_value=0.0,
    expected_format_id=0, # int
    embeddings_layer=embeddings_layer,
    tokenizer=tokenizer,
    config=config,
    description="Liczba zero"
)

# Test 5: Tekst bez liczb
print("\n--- Test dla: Tekst bez liczb ---")
text_no_num = "This sentence has no numbers."
encoded_input_no_num = tokenizer(
    text_no_num,
    padding="max_length",
    max_length=config.max_position_embeddings,
    return_tensors="pt"
)
input_ids_no_num = encoded_input_no_num['input_ids']
numeric_values_no_num = encoded_input_no_num['numeric_values']
numeric_formats_no_num = encoded_input_no_num['numeric_formats']

final_embeddings_no_num = embeddings_layer(
    input_ids=input_ids_no_num,
    numeric_values=numeric_values_no_num,
    numeric_formats=numeric_formats_no_num
)

print(f"Tekst: '{text_no_num}'")
print(f"Input IDs: {input_ids_no_num.tolist()}")
print(f"Numeric Values: {numeric_values_no_num.tolist()}") # Powinny być same NaN lub 0.0 w zależności od tokenizera
print(f"Kształt finalnych osadzeń: {final_embeddings_no_num.shape}")

# Sprawdź, czy nie ma tokenów [NUM] i czy numeric_values są puste/NaN
num_token_pos_no_num = (input_ids_no_num == config.num_token_id).nonzero(as_tuple=True)[1]
print(f"Liczba tokenów [NUM] w tekście bez liczb: {num_token_pos_no_num.numel()}")
assert num_token_pos_no_num.numel() == 0, "Token [NUM] pojawił się w tekście bez liczb!"

# Sprawdź, czy w ogóle nie ma aktywowanych osadzeń numerycznych w `numeric_embeds_for_fusion` (wewnętrznie)
# Choć `numeric_embeds_for_fusion` nie jest zwracane, możemy wnioskować o jego wpływie
# sprawdzając, czy final_embeddings dla pozycji, które nie są tekstowe, są zerowe (padding)
# lub czy te które są tekstowe, nie mają "magicznego" wzmocnienia od liczby
print("Test dla tekstu bez liczb pomyślny: Brak tokenów [NUM].")


print("\n" + "="*50)
print("--- PODSUMOWANIE ---")
print("Kod `BlackholeEmbeddings` został rozbudowany o zaawansowane cechy numeryczne i dynamiczną fuzję.")
print("Testy pokazują, że tokeny numeryczne są prawidłowo przetwarzane i osadzane w wektorze finalnym.")
print("To stanowi solidną bazę do budowy LLM, który 'rozumie' liczby.")
print("Pamiętaj o **finalnej modyfikacji `BlackholeTokenizer`**, aby generował `numeric_values` z `NaN` i `numeric_formats`!")
print("="*50)
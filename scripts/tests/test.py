import sys
import os
import re

# Upewnij się, że ścieżka do Twojego katalogu 'blackhole' jest poprawna.
# To jest KLUCZOWE, aby BlackholeTokenizer był dostępny.
try:
    # Zakładając, że skrypt jest w podobnej strukturze jak w przykładzie USEME
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from blackhole.tokenizer_hugging_face import BlackholeTokenizer
except ImportError:
    print("Błąd: Nie można zaimportować BlackholeTokenizer.")
    print("Upewnij się, że biblioteka 'blackhole-llm' jest zainstalowana i poprawnie zaimportowana.")
    print("Może być konieczne dostosowanie ścieżki w 'sys.path.insert'.")
    # Zakończ działanie skryptu, jeśli import się nie powiedzie
    sys.exit(1)

def extract_and_sum_numbers_with_blackhole(text_to_process, tokenizer_output_dir="./my_custom_tokenizer_test"):
    """
    Używa BlackholeTokenizer do przetworzenia tekstu, a następnie
    wyodrębnia liczby na podstawie specjalnego tokena [NUM] w dekodowanym tekście,
    określa ich typ (liczbowy czy string) i oblicza sumę.

    Args:
        text_to_process (str): Tekst do przetworzenia.
        tokenizer_output_dir (str): Katalog, w którym zapisano/załadowano tokenizer.
    """
    print(f"--- Przetwarzanie tekstu: '{text_to_process}' ---")

    # 1. Załadowanie Tokenizera (lub inicjalizacja i trening, jeśli jeszcze nie masz)
    # W praktyce, zazwyczaj ładujesz wytrenowany tokenizer
    try:
        if not os.path.exists(tokenizer_output_dir) or not any(f.endswith('.json') for f in os.listdir(tokenizer_output_dir)):
            print(f"Katalog tokenizera '{tokenizer_output_dir}' nie istnieje lub jest pusty.")
            print("Inicjowanie i trenowanie tokenizera na próbkowych danych...")
            tokenizer = BlackholeTokenizer()
            # Użyj przykładowych tekstów z Twojego USEME do treningu
            texts_for_training = [
                "Hello world! This is a test. The number is 123.45 and also 0xabc.",
                "Another EXAMPLE sentence with DATE 2023-10-26 and time 14:30. What about i.e. and e.g.?",
                "Numbers: +1000, -5.5e-2, 999,999.00. Operators: ->, <=, ==.",
                "ALL CAPS TEXT. First Capital Letter.",
                "Unicode hyphen: this–that. At-tag: @xmath0. A sentence with ellipsis... and quotes 'like this'."
            ]
            tokenizer.train_tokenizer(texts_for_training, vocab_size=8000, min_freq=1)
            os.makedirs(tokenizer_output_dir, exist_ok=True)
            tokenizer.save_pretrained(tokenizer_output_dir)
            print(f"Tokenizer wytrenowany i zapisany do: {tokenizer_output_dir}")
        else:
            print(f"Ładowanie tokenizera z: {tokenizer_output_dir}")
            tokenizer = BlackholeTokenizer.from_pretrained(tokenizer_output_dir)

    except Exception as e:
        print(f"Błąd podczas ładowania/trenowania tokenizera: {e}")
        return

    # 2. Zakodowanie i Dekodowanie tekstu
    # Zwracamy input_ids, ponieważ to one zawierają informację o numerach
    encoded_input = tokenizer(text_to_process, return_tensors="pt", padding=True, truncation=False)
    # Dekodujemy z uwzględnieniem tokenów specjalnych, aby zobaczyć [NUM]
    decoded_with_special_tokens = tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=False)

    print(f"\nDekodowany tekst (z tokenami specjalnymi): '{decoded_with_special_tokens}'")

    tokens = re.split(r'(\[NUM\]|\[CAP\]|\[ALLCAPS\]|\s+)', decoded_with_special_tokens)
    
    found_numbers = []
    # Iterujemy po tokenach, szukając [NUM], a następnie kolejnego tokena, który powinien być liczbą
    i = 0
    while i < len(tokens):
        if tokens[i] == '[NUM]' and i + 1 < len(tokens):
            potential_number_str = tokens[i+1].strip()
            # Regex do walidacji, czy to rzeczywiście liczba
            number_regex = r"^[+-]?\b(?:(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|0x[0-9a-fA-F]+)\b$"
            if re.match(number_regex, potential_number_str):
                found_numbers.append(potential_number_str)
            i += 2 # Przesuwamy się o 2, bo skonsumowaliśmy [NUM] i liczbę
        else:
            i += 1
            
    extracted_numerical_values = []
    print("\n--- Znalezione liczby i ich format ---")
    if not found_numbers:
        print("Nie znaleziono żadnych liczb oznaczonych jako [NUM].")
    else:
        for num_str in found_numbers:
            original_num_str = num_str # Zachowujemy oryginalny string do wyświetlenia
            num_type = "String"
            num_value = None

            try:
                if re.match(r"0x[0-9a-fA-F]+", num_str):
                    num_value = int(num_str, 16)
                    num_type = "Integer (from hexadecimal)"
                elif '.' in num_str or 'e' in num_str.lower():
                    clean_num_str = num_str.replace(',', '')
                    num_value = float(clean_num_str)
                    num_type = "Float"
                else:
                    clean_num_str = num_str.replace(',', '')
                    num_value = int(clean_num_str)
                    num_type = "Integer"
                extracted_numerical_values.append(num_value)
                print(f"'{original_num_str}': {num_type} - Wartość: {num_value}")
            except ValueError:
                print(f"'{original_num_str}': {num_type} (nie udało się przekonwertować na liczbę)")
            except Exception as e:
                print(f"Błąd przetwarzania '{original_num_str}': {e}")


    # 4. Operacje na liczbach
    print("\n--- Operacje na liczbach ---")
    if extracted_numerical_values:
        total_sum = sum(extracted_numerical_values)
        print(f"Liczby dodane: {extracted_numerical_values}")
        print(f"Suma wszystkich znalezionych liczb: {total_sum}")
    else:
        print("Brak liczb do wykonania operacji.")

# --- Przykładowe użycie ---
sample_text_1 = "This is an example text. The value is 1,234.56. ALL CAPS! Another word. @xcite_here"
sample_text_2 = "Test with more numbers: 100, -25.5, 0xABC, 1.23e-4, 999. What about 2023-10-26?"
sample_text_3 = "No numbers here, just a plain sentence."

# Uruchomienie funkcji dla przykładowego tekstu
extract_and_sum_numbers_with_blackhole(sample_text_1)
print("\n" + "="*80 + "\n") # Separator dla czytelności
extract_and_sum_numbers_with_blackhole(sample_text_2)
print("\n" + "="*80 + "\n")
extract_and_sum_numbers_with_blackhole(sample_text_3)
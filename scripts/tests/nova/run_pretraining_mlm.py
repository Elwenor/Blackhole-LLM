import torch
import os, sys
import math
from transformers import logging, set_seed
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Union
from datasets import load_dataset, Dataset as HFDataset # Importuj Dataset z Hugging Face

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..\..')))


from blackhole.nova_hugging_face_encoder.modeling_nova import BlackholeForMaskedLM, BlackholeConfig
from blackhole.nova_hugging_face_encoder import NovaTrainer, TrainingArguments # Poprawiona ścieżka importu
from blackhole.tokenizer_hugging_face import BlackholeTokenizer
from blackhole.nova_hugging_face_encoder import BlackholeDataCollatorForLanguageModeling # Upewnij się, że ten import jest poprawny
from blackhole.tokenizer_hugging_face import CUSTOM_SPECIAL_TOKENS

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

# --- Nowa Definicja Datasetu ---
# Ta klasa TextNumericDataset teraz bezpośrednio przetwarza dane z obiektu Hugging Face Dataset.
class TextNumericDataset(Dataset):
    def __init__(self, 
                 tokenizer: BlackholeTokenizer, 
                 hf_dataset: HFDataset, # Przyjmuje obiekt Hugging Face Dataset
                 max_length: int,
                 sample_percentage: float = 1.0): # Dodajemy procent próbkowania
        
        self.tokenizer = tokenizer
        self.hf_dataset = hf_dataset
        self.max_length = max_length
        self.examples = []

        logger.info(f"Wczytuję dane z Hugging Face Dataset. Rozmiar oryginalny: {len(hf_dataset)} przykładów.")
        
        # Ogranicz liczbę przykładów na podstawie sample_percentage
        num_samples_to_load = int(len(hf_dataset) * sample_percentage)
        logger.info(f"Będzie przetworzonych {num_samples_to_load} ({sample_percentage*100:.2f}%) przykładów.")

        for i, item in enumerate(hf_dataset):
            if i >= num_samples_to_load:
                break
            
            question = item['question']
            answer = item['answer'] # Pełne rozwiązanie z odpowiedziami
            
            # Połącz pytanie i rozwiązanie, aby uzyskać więcej kontekstu numerycznego
            full_text = f"{question} {answer}"

            # Zakładamy, że tokenizer automatycznie przetwarza liczby na [NUM]
            # i generuje odpowiednie numeric_values oraz numeric_formats
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_numeric_features=True # To jest kluczowe dla Twojego tokenizatora
            )
            
            # Upewnij się, że tensor jest 1D dla dataloadera
            self.examples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "numeric_values": encoding["numeric_values"].squeeze(0),
                "numeric_formats": encoding["numeric_formats"].squeeze(0),
            })
        
        logger.info(f"Liczba przykładów (ostatecznie użytych w dataset): {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.examples[idx]

# --- Główna funkcja treningowa ---
def main():
    # 1. Konfiguracja i Inicjalizacja Tokenizatora
    # -----------------------------------------------
    print("================================================================================")
    print("     --- 1. Blackhole Tokenizer Configuration and Initialization ---")
    print("================================================================================")

    tokenizer_path = "./blackhole_tokenizer_demo"
    if os.path.exists(tokenizer_path):
        print(f"Tokenizator znaleziony w '{tokenizer_path}'. Wczytuję istniejący.")
        tokenizer = BlackholeTokenizer.from_pretrained(tokenizer_path)
    else:
        print(f"Tokenizator nie znaleziony w '{tokenizer_path}'. Tworzę nowy.")
        tokenizer = BlackholeTokenizer.from_pretrained(
            "allegro/herbert-base-cased", # Możesz zmienić bazowy model
            do_lower_case=False
        )
        # Dodaj specjalne tokeny do tokenizatora
        num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': list(CUSTOM_SPECIAL_TOKENS.values())})
        if num_added_tokens > 0:
            logger.info(f"Dodano {num_added_tokens} nowych specjalnych tokenów do tokenizatora.")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Nowy tokenizator zapisany w '{tokenizer_path}'")
    
    print(f"-> ID dla specjalnego tokena [NUM]: {tokenizer.convert_tokens_to_ids(CUSTOM_SPECIAL_TOKENS['number_token'])}")
    print(f"-> ID dla specjalnego tokena [MASK]: {tokenizer.convert_tokens_to_ids(CUSTOM_SPECIAL_TOKENS['mask_token'])}")
    print(f"Ostateczny rozmiar słownika tokenizatora: {len(tokenizer)}") # Wypisz ostateczny rozmiar słownika

    # 2. Inicjalizacja Modelu
    # -----------------------------------------------
    # WAŻNE: vocab_size w konfiguracji modelu MUSI być zgodne z len(tokenizer)
    config = BlackholeConfig(
        vocab_size=len(tokenizer), # KLUCZOWE: Użyj rozmiaru słownika z tokenizatora
        max_position_embeddings=128, # Zwiększ, jeśli potrzebujesz dłuższych sekwencji
        num_attention_heads=4,
        num_hidden_layers=3,
        type_vocab_size=1,
        hidden_size=256, 
        intermediate_size=512, 
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        num_token_id=tokenizer.convert_tokens_to_ids(CUSTOM_SPECIAL_TOKENS["number_token"]), # Upewnij się, że to jest ustawione
    )
    model = BlackholeForMaskedLM(config)

    # Jeśli tokenizator został zmodyfikowany (np. dodano tokeny) PO inicjalizacji modelu,
    # należy zmienić rozmiar embeddingów modelu.
    # W tym przypadku, jeśli tokenizator jest tworzony od nowa i dodajemy tokeny,
    # to model jest tworzony z prawidłowym vocab_size, więc resize_token_embeddings
    # nie jest bezwzględnie konieczne, ale nie zaszkodzi dla pewności.
    # Jeśli ładujesz istniejący tokenizator i dodajesz tokeny, to jest to KRYTYCZNE.
    # W tym przypadku, jeśli blok 'else' jest uruchamiany, to `num_added_tokens` będzie > 0.
    # Jeśli blok 'if' jest uruchamiany (tokenizator już istnieje), to `num_added_tokens` będzie 0.
    # Możesz to uprościć, wywołując resize_token_embeddings zawsze po inicjalizacji modelu,
    # jeśli len(tokenizer) != model.config.vocab_size
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Zmieniono rozmiar embeddingów modelu na {len(tokenizer)}.")

    print(f"Liczba trenowalnych parametrów modelu: {model.num_parameters()}")
    print(f"Rozmiar słownika modelu (config.vocab_size): {model.config.vocab_size}")


    # 3. Argumenty Treningowe
    # -----------------------------------------------
    training_args = TrainingArguments(
        output_dir="./blackhole_mlm_output",
        per_device_train_batch_size=8, # Zwiększono batch size
        per_device_eval_batch_size=8,
        num_train_epochs=3.0, # Zwiększono liczbę epok, możesz potrzebować więcej
        learning_rate=5e-5,
        logging_steps=100, 
        eval_steps=500,
        save_steps=500,
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(), 
        seed=42,
    )
    set_seed(training_args.seed)

    # 4. Ładowanie i przygotowanie danych z GSM8K
    # -----------------------------------------------
    print("\n================================================================================")
    print("     --- 4. Ładowanie danych z GSM8K ---")
    print("================================================================================")
    
    # Pobierz GSM8K
    try:
        gsm8k_dataset = load_dataset("openai/gsm8k", "main")
        print(f"Pomyślnie wczytano dataset GSM8K. Dostępne splity: {gsm8k_dataset.keys()}")
    except Exception as e:
        logger.error(f"Nie udało się wczytać GSM8K: {e}. Upewnij się, że masz połączenie z internetem.")
        return # Zakończ, jeśli nie można wczytać danych

    # Zdefiniuj procent danych do użycia
    # Zalecane: Zacznij od mniejszego procentu (np. 0.1) dla szybkich testów,
    # a potem zwiększaj do 1.0 dla pełnego treningu.
    TRAIN_DATA_PERCENTAGE = 0.5 # Użyj 50% danych treningowych GSM8K
    EVAL_DATA_PERCENTAGE = 1.0  # Użyj 100% danych walidacyjnych GSM8K

    # Utwórz instancje datasetów dla Twojego Trainera
    train_dataset = TextNumericDataset(
        tokenizer, 
        gsm8k_dataset['train'], 
        config.max_position_embeddings,
        sample_percentage=TRAIN_DATA_PERCENTAGE
    )
    eval_dataset = TextNumericDataset(
        tokenizer, 
        gsm8k_dataset['test'], # GSM8K używa 'test' jako splitu ewaluacyjnego
        config.max_position_embeddings,
        sample_percentage=EVAL_DATA_PERCENTAGE
    )

    # 5. Inicjalizacja DataCollatora
    # -----------------------------------------------
    data_collator = BlackholeDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=8,
        seed=training_args.seed,
        max_length=config.max_position_embeddings # Ważne: max_length dla collatora
    )

    # 6. Inicjalizacja i uruchomienie Trainera
    # -----------------------------------------------
    print("\n================================================================================")
    print("     --- Rozpoczynanie Treningu Modelu Blackhole ---")
    print("================================================================================")

    trainer = NovaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    print("\n================================================================================")
    print("             --- Trening Zakończony ---")
    print(f"Model i tokenizator zostały zapisane w: {training_args.output_dir}")
    print("================================================================================")

    # --- Testowanie Wczytanego Modelu ---
    # -----------------------------------------------
    print("\n--- Testowanie Wczytanego Modelu ---")
    final_model_path = os.path.join(training_args.output_dir, "final")
    
    # Wczytaj model i tokenizator z finalnego checkpointu
    loaded_model = BlackholeForMaskedLM.from_pretrained(final_model_path)
    loaded_tokenizer = BlackholeTokenizer.from_pretrained(final_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    loaded_model.eval()

    # Przykładowe zdanie do testów
    sentence = 'The initial measurement was 123.45. After several trials, we recorded new data points: 0.007 and then -100. Furthermore, the final sum was 5.0e-3. The experiment concluded with 99.9% accuracy and a total of 1000 iterations over 25.5 hours.'
    
    # Tokenizujemy zdanie, zakładając, że tokenizer sam zamienia liczby na [NUM]
    # i zwraca numeric_values oraz numeric_formats
    inputs = loaded_tokenizer(
        sentence,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=loaded_model.config.max_position_embeddings,
        return_numeric_features=True
    )

    # Znajdź indeksy tokenów [NUM]
    num_token_id = loaded_tokenizer.convert_tokens_to_ids(CUSTOM_SPECIAL_TOKENS["number_token"])
    mask_token_id = loaded_tokenizer.convert_tokens_to_ids(CUSTOM_SPECIAL_TOKENS["mask_token"])

    masked_num_indices = (inputs['input_ids'] == num_token_id).nonzero(as_tuple=True)

    # Stwórz zamaskowane inputy dla testu MLM
    mlm_input_ids = inputs['input_ids'].clone()
    mlm_attention_mask = inputs['attention_mask'].clone()
    mlm_numeric_values = inputs['numeric_values'].clone()
    mlm_numeric_formats = inputs['numeric_formats'].clone()

    # Maskuj znalezione tokeny [NUM]
    for idx in masked_num_indices[1]: 
        mlm_input_ids[0, idx] = mask_token_id # Zamień [NUM] na [MASK]
    
    # Przenieś dane na odpowiednie urządzenie
    mlm_inputs = {
        'input_ids': mlm_input_ids.to(device),
        'attention_mask': mlm_attention_mask.to(device),
        'numeric_values': mlm_numeric_values.to(device),
        'numeric_formats': mlm_numeric_formats.to(device)
    }

    print(f"Testowanie zdania: '{sentence}'")
    print(f"Tokenizowane input_ids (z zamaskowanymi [NUM]): {mlm_inputs['input_ids']}")
    print(f"Pozycje zamaskowanych [NUM] (indeksy w input_ids): {masked_num_indices[1].tolist()}")

    with torch.no_grad():
        outputs = loaded_model(**mlm_inputs)
        logits = outputs.logits 

    print(f"Kształt logitów: {logits.shape}")

    # Przewidywanie top-K dla każdej zamaskowanej pozycji [NUM]
    top_k = 10 

    for i, mask_idx in enumerate(masked_num_indices[1]):
        # Uzyskaj logity dla zamaskowanej pozycji
        mask_logits = logits[0, mask_idx, :] 

        # Uzyskaj top-K wartości i indeksów
        top_k_values, top_k_indices = torch.topk(mask_logits, top_k)

        original_num_value = inputs['numeric_values'][0, mask_idx].item()
        original_num_format = inputs['numeric_formats'][0, mask_idx].item()

        print(f"\n--- Przewidywania dla [NUM] na pozycji {mask_idx} ---")
        print(f"Oryginalna liczba (wartość numeryczna): {original_num_value}")
        print(f"Oryginalny format (numeryczny): {original_num_format}")
        
        for k in range(top_k):
            predicted_token_id = top_k_indices[k].item()
            predicted_token_score = top_k_values[k].item()
            predicted_token = loaded_tokenizer.convert_ids_to_tokens(predicted_token_id)
            print(f"   {k+1}. Token: '{predicted_token}' (ID: {predicted_token_id}), Logit: {predicted_token_score:.4f}")

    print("\n--- Prosta demonstracja działania po treningu zakończona ---")


if __name__ == '__main__':
    main()

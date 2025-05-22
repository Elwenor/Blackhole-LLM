import sys, os
import torch
from typing import List, Dict, Any, Tuple, Optional, Union, Iterator

# Dodaj katalog zawierający hugging_tokenizer.py do ścieżki Pythona
# Zakładamy, że ten skrypt jest w tym samym katalogu co hugging_tokenizer.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from hugging_tokenizer import MyCustomTokenizer # Zaktualizowana klasa MyCustomTokenizer

import tokenizers
print(f"Tokenizers version: {tokenizers.__version__}")

def print_test_results(title, original_text, encoded_input, decoded_with_special, decoded_clean, tokenizer_obj):
    """Pomocnicza funkcja do drukowania wyników testów w czytelny sposób."""
    
    # encoded_input to BatchEncoding, wyciągnij pierwszy element dla testów pojedynczych
    # input_ids i attention_mask mogą być tensorami lub listami list
    
    # Upewnij się, że input_ids_list jest zawsze płaską listą liczb całkowitych
    if isinstance(encoded_input['input_ids'], torch.Tensor):
        # Jeśli to tensor, konwertuj na listę. Jeśli to partia, weź pierwszy element.
        if encoded_input['input_ids'].ndim > 1:
            encoded_ids_list = encoded_input['input_ids'][0].tolist()
        else:
            encoded_ids_list = encoded_input['input_ids'].tolist()
    elif isinstance(encoded_input['input_ids'], list):
        # Jeśli to lista, sprawdź, czy to lista list (partia) czy płaska lista
        if encoded_input['input_ids'] and isinstance(encoded_input['input_ids'][0], list):
            encoded_ids_list = encoded_input['input_ids'][0] # Weź pierwszą sekwencję
        else:
            encoded_ids_list = encoded_input['input_ids'] # Jest już płaską listą
    else:
        # Fallback dla nieoczekiwanych typów, choć BatchEncoding powinien obejmować tensory/listy
        encoded_ids_list = [] # Lub zgłoś błąd

    # Upewnij się, że attention_mask_list jest zawsze płaską listą liczb całkowitych
    if isinstance(encoded_input['attention_mask'], torch.Tensor):
        if encoded_input['attention_mask'].ndim > 1:
            attention_mask_list = encoded_input['attention_mask'][0].tolist()
        else:
            attention_mask_list = encoded_input['attention_mask'].tolist()
    elif isinstance(encoded_input['attention_mask'], list):
        if encoded_input['attention_mask'] and isinstance(encoded_input['attention_mask'][0], list):
            attention_mask_list = encoded_input['attention_mask'][0]
        else:
            attention_mask_list = encoded_input['attention_mask']
    else:
        attention_mask_list = []

    # --- Poprawne użycie convert_ids_to_tokens ---
    try:
        # `convert_ids_to_tokens` przyjmuje listę ID i zwraca listę stringów
        # To jest metoda z PreTrainedTokenizerFast i powinna działać z płaską listą intów.
        encoded_tokens_from_ids = tokenizer_obj.convert_ids_to_tokens(encoded_ids_list)
    except Exception as e:
        encoded_tokens_from_ids = f"Error converting IDs to tokens: {e}"

    print(f"\n--- Test: {title} ---")
    print(f"Original text:                 '{original_text}'")
    print(f"Encoded input (IDs):           {encoded_ids_list}") 
    print(f"Encoded tokens (from IDs):     {encoded_tokens_from_ids}") 
    print(f"Attention mask:                {attention_mask_list}")
    
    # Dostęp do przechowywanych metadanych i kodowania dla tej partii
    current_encoding = tokenizer_obj._last_encodings_objects[0] if tokenizer_obj._last_encodings_objects else None
    
    current_metadata_tuple = tokenizer_obj._last_original_metadata_for_decode[0] if tokenizer_obj._last_original_metadata_for_decode else ([], [])
    original_word_metadata_list_for_trace = current_metadata_tuple[0]
    map_processed_idx_to_original_meta_idx_for_trace = current_metadata_tuple[1]

    print("\n--- Detailed Tokenization Trace ---")
    print("Original Pre-tokenized Unit | Metadata Type | BPE Token ID | BPE Token Str | Word ID (from Encoding)")
    print("-----------------------------------------------------------------------------------------------------")

    if current_encoding:
        for bpe_token_idx in range(len(current_encoding.ids)):
            bpe_id = current_encoding.ids[bpe_token_idx]
            
            if isinstance(bpe_id, int):
                bpe_token_str = tokenizer_obj.convert_ids_to_tokens([bpe_id])[0]
            else:
                bpe_token_str = f"Invalid ID: {bpe_id}"

            word_id_from_encoding = current_encoding.word_ids[bpe_token_idx] if current_encoding.word_ids else None
            
            original_pretoken_unit = "N/A"
            metadata_type = "N/A"
            
            if word_id_from_encoding is not None and word_id_from_encoding < len(map_processed_idx_to_original_meta_idx_for_trace):
                original_meta_idx = map_processed_idx_to_original_meta_idx_for_trace[word_id_from_encoding]
                if original_meta_idx is not None and original_meta_idx < len(original_word_metadata_list_for_trace):
                    original_pretoken_unit = original_word_metadata_list_for_trace[original_meta_idx]['original_value']
                    metadata_type = original_word_metadata_list_for_trace[original_meta_idx]['type']

            print(f"{original_pretoken_unit:<27} | {metadata_type:<13} | {bpe_id:<12} | {bpe_token_str:<13} | {str(word_id_from_encoding):<25}")
    else:
        print("No encoding object or metadata available for detailed trace (likely a batch > 1 or error).")
    
    print("-----------------------------------------------------------------------------------------------------")

    print(f"Decoded text (with special):   '{decoded_with_special}'")
    print(f"Decoded text (without special):'{decoded_clean}'")
    
    if original_text.strip() != decoded_clean.strip():
        print(f"MISMATCH: Original and decoded text (clean) are different.")
    else:
        print(f"MATCH: Original and decoded text (clean) are identical.")

if __name__ == "__main__":
    tokenizer = MyCustomTokenizer()

    texts_for_training = [
        "Hello world! This is a test. The number is 123.45 and also 0xabc.",
        "Another EXAMPLE sentence with DATE 2023-10-26 and time 14:30. What about i.e. and e.g.?",
        "Numbers: +1000, -5.5e-2, 999,999.00. Operators: ->, <=, ==.",
        "ALL CAPS TEXT. First Capital Letter.",
        "Unicode hyphen: this–that. At-tag: @xmath0. A sentence with ellipsis... and quotes 'like this'.",
        "It's a wonderful day. (Hello) [World] {Python}.",
        "Testing special characters: #hash $dollar %percent ^caret &amp *star (open) [bracket] {brace}.",
        "Negative number: -100. Range: 5-10. Word-word. Well-being and T-shirt. ID345-ABC.",
        "This is a long sentence that should not be truncated.",
        "Words with hyphens: well-being, T-shirt, state-of-the-art.",
        "Dates and times in different formats: 2023/12/31, 09:00:00. Hexadecimal: 0xDEADBEEF.",
        "Mixed case and numbers like ID345-ABC.",
        "URL: https://example.com/page?id=123&name=test. Email: user@domain.com.",
        "Code snippets: variable_name = 10; function_call(arg1, arg2);",
        "Complex numbers: 3+4i, -2.5-1.2j.",
        "Scientific notation: 6.022e23.",
        "Currency: $100.00, €50, ¥200.",
        "Abbreviations: U.S.A., Ph.D., Dr. No.",
        "Smileys and emojis: :) :( :D 👍 🚀",
        "Special symbols: © ® ™ § † ‡ ∞ ∑",
        "More punctuation: 'Hello world.' \"Hello world.\" [hello world] {hello world}",
        "A mix of everything: The result is -1.23e-4. (See page 12-15). @user1 #topic. Is it 2024/07/21? Yes!",
        "Sentence with multiple spaces    and    tabs    for    testing."
    ]

    print("Training tokenizer...")
    tokenizer.train_tokenizer(texts_for_training, vocab_size=8000, min_freq=1) 
    print(f"Vocab size after training: {tokenizer.vocab_size}")

    output_dir = "./my_custom_tokenizer_test"
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    loaded_tokenizer = MyCustomTokenizer.from_pretrained(output_dir)
    print(f"Tokenizer loaded from {output_dir}")

    # --- Test 1: Standard Text with Numbers and Caps ---
    test_text_1 = "This is an example text. The value is 1,234.56. ALL CAPS! Another word. @xcite_here"
    encoded_input_1 = loaded_tokenizer(test_text_1, return_tensors="pt", padding=True, truncation=False)
    decoded_text_1 = loaded_tokenizer.decode(encoded_input_1['input_ids'][0], skip_special_tokens=False)
    decoded_clean_1 = loaded_tokenizer.decode(encoded_input_1['input_ids'][0], skip_special_tokens=True)
    print_test_results("Standard Text with Numbers and Caps", test_text_1, encoded_input_1, decoded_text_1, decoded_clean_1, loaded_tokenizer)

    # --- Test 2: All Caps and Initial Caps ---
    test_text_2 = "THIS IS ALL CAPS. This Is Initial Caps. normal word."
    encoded_input_2 = loaded_tokenizer(test_text_2, return_tensors="pt")
    decoded_text_2 = loaded_tokenizer.decode(encoded_input_2['input_ids'][0], skip_special_tokens=False)
    decoded_clean_2 = loaded_tokenizer.decode(encoded_input_2['input_ids'][0], skip_special_tokens=True)
    print_test_results("All Caps and Initial Caps", test_text_2, encoded_input_2, decoded_text_2, decoded_clean_2, loaded_tokenizer)

    # --- Test 3: Punctuation and Operators ---
    test_text_3 = "Numbers: +1000, -5.5e-2, 999,999.00. Operators: ->, <=, ==. This (is) [a] {test}."
    encoded_input_3 = loaded_tokenizer(test_text_3, return_tensors="pt")
    decoded_text_3 = loaded_tokenizer.decode(encoded_input_3['input_ids'][0], skip_special_tokens=False)
    decoded_clean_3 = loaded_tokenizer.decode(encoded_input_3['input_ids'][0], skip_special_tokens=True)
    print_test_results("Punctuation and Operators", test_text_3, encoded_input_3, decoded_text_3, decoded_clean_3, loaded_tokenizer)

    # --- Test 4: Unicode Hyphens and Special Characters ---
    test_text_4 = "Unicode hyphen: this–that. At-tag: @xmath0. A sentence with ellipsis... and quotes 'like this'."
    encoded_input_4 = loaded_tokenizer(test_text_4, return_tensors="pt")
    decoded_text_4 = loaded_tokenizer.decode(encoded_input_4['input_ids'][0], skip_special_tokens=False)
    decoded_clean_4 = loaded_tokenizer.decode(encoded_input_4['input_ids'][0], skip_special_tokens=True)
    print_test_results("Unicode Hyphens and Special Characters", test_text_4, encoded_input_4, decoded_text_4, decoded_clean_4, loaded_tokenizer)

    # --- Test 5: Dates and Times (Subword Tokenization) ---
    test_text_5 = "Meeting on 2023-11-15 at 10:00. Project due by 2024-01-01 23:59:59."
    encoded_input_5 = loaded_tokenizer(test_text_5, return_tensors="pt")
    decoded_text_5 = loaded_tokenizer.decode(encoded_input_5['input_ids'][0], skip_special_tokens=False)
    decoded_clean_5 = loaded_tokenizer.decode(encoded_input_5['input_ids'][0], skip_special_tokens=True)
    print_test_results("Dates and Times (Subword Tokenization)", test_text_5, encoded_input_5, decoded_text_5, decoded_clean_5, loaded_tokenizer)

    # --- Test 6: Whitespace Handling ---
    test_text_6 = "    Hello    World!     This is a test.    "
    encoded_input_6 = loaded_tokenizer(test_text_6, return_tensors="pt")
    decoded_text_6 = loaded_tokenizer.decode(encoded_input_6['input_ids'][0], skip_special_tokens=False)
    decoded_clean_6 = loaded_tokenizer.decode(encoded_input_6['input_ids'][0], skip_special_tokens=True)
    print_test_results("Whitespace Handling", test_text_6, encoded_input_6, decoded_text_6, decoded_clean_6, loaded_tokenizer)

    # --- Test 7: UNK Token Handling (Expected Subword Fallback) ---
    test_text_7 = "This is a brand new_word. And some_other_word. Also a SuperMegaWord."
    encoded_input_7 = loaded_tokenizer(test_text_7, return_tensors="pt")
    decoded_text_7 = loaded_tokenizer.decode(encoded_input_7['input_ids'][0], skip_special_tokens=False)
    decoded_clean_7 = loaded_tokenizer.decode(encoded_input_7['input_ids'][0], skip_special_tokens=True)
    print_test_results("UNK Token Handling (Expected Subword Fallback)", test_text_7, encoded_input_7, decoded_text_7, decoded_clean_7, loaded_tokenizer)

    # --- Test 8: Hyphens in Words and Numbers ---
    test_text_8 = "Negative number: -100. Range: 5-10. Word-word. Well-being and T-shirt. ID345-ABC."
    encoded_input_8 = loaded_tokenizer(test_text_8, return_tensors="pt")
    decoded_text_8 = loaded_tokenizer.decode(encoded_input_8['input_ids'][0], skip_special_tokens=False)
    decoded_clean_8 = loaded_tokenizer.decode(encoded_input_8['input_ids'][0], skip_special_tokens=True)
    print_test_results("Hyphens in Words and Numbers", test_text_8, encoded_input_8, decoded_text_8, decoded_clean_8, loaded_tokenizer)

    # --- Test 9: URLs and Emails ---
    test_text_9 = "Visit https://example.com/page?id=123. Email: user@domain.com."
    encoded_input_9 = loaded_tokenizer(test_text_9, return_tensors="pt")
    decoded_text_9 = loaded_tokenizer.decode(encoded_input_9['input_ids'][0], skip_special_tokens=False)
    decoded_clean_9 = loaded_tokenizer.decode(encoded_input_9['input_ids'][0], skip_special_tokens=True)
    print_test_results("URLs and Emails", test_text_9, encoded_input_9, decoded_text_9, decoded_clean_9, loaded_tokenizer)

    # --- Test 10: Special symbols ---
    test_text_10 = "Copyright ©. Registered ®. Trademark ™. Section §. Dagger †. Double Dagger ‡. Infinity ∞. Sum ∑."
    encoded_input_10 = loaded_tokenizer(test_text_10, return_tensors="pt")
    decoded_text_10 = loaded_tokenizer.decode(encoded_input_10['input_ids'][0], skip_special_tokens=False)
    decoded_clean_10 = loaded_tokenizer.decode(encoded_input_10['input_ids'][0], skip_special_tokens=True)
    print_test_results("Special Symbols", test_text_10, encoded_input_10, decoded_text_10, decoded_clean_10, loaded_tokenizer)

    # --- Test 11: Currency ---
    test_text_11 = "Price is $100.00, also €50 and ¥200."
    encoded_input_11 = loaded_tokenizer(test_text_11, return_tensors="pt")
    decoded_text_11 = loaded_tokenizer.decode(encoded_input_11['input_ids'][0], skip_special_tokens=False)
    decoded_clean_11 = loaded_tokenizer.decode(encoded_input_11['input_ids'][0], skip_special_tokens=True)
    print_test_results("Currency", test_text_11, encoded_input_11, decoded_text_11, decoded_clean_11, loaded_tokenizer)


    # Informacje o tokenach specjalnych
    print(f"\n--- Special Token Information ---")
    print(f"Special tokens: {loaded_tokenizer.all_special_tokens}")
    print(f"Num token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.num_token)}")
    print(f"Cap token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.cap_token)}")
    print(f"Allcaps token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.allcaps_token)}")
    print(f"UNK token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.unk_token)}") 
    print(f"CLS token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.cls_token)}") 
    print(f"PAD token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.pad_token)}") 
    print(f"SEP token ID: {loaded_tokenizer.vocab.get(loaded_tokenizer.sep_token)}")

import re
from collections import Counter, OrderedDict
import unicodedata
import torch
import torch.nn as nn
import math
import struct
import os # Dodajemy do zarządzania cache

# --- TOKENIZER.PY (lub miejsce, gdzie trzymasz tokenizer) ---

# Definicje wzorców
hex_pattern_str = r'0x[0-9A-Fa-f]+'
number_pattern_str = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?'
date_pattern_str = r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
time_pattern_str = r'\b\d{1,2}:\d{2}(?::\d{2})?\b'

tokenizer_pattern_re = re.compile(
    r'(?:@[\w]+)|'                      # @xcite, @xmath0 itp.
    r'(?:e\.g\.)|'                      # "e.g." w całości
    r'(?:i\.e\.)|'                      # "i\.e\." w całości
    r'(?:0x[0-9a-fA-F]+)|'              # hex
    r'(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})|' # daty
    r'(?:\d{1,2}:\d{2}(?::\d{2})?)|'    # czasy
    r'(?:[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)|' # liczby
    r'(?:\.\.\.)|'                      # Ellipsis (...) - WAŻNE: musi być przed pojedynczymi kropkami
    r'(?:->|<-|==|!=|>=|<=|>>|<<|::=|--)|' # Złożone operatory/sekwencje symboli
    r'(?:[A-Za-z]+(?:[\'-][A-Za-z]+)*)|' # słowa (w tym z myślnikiem lub apostrofem np. "don't", "co-worker")
    r'(?:[\u2010-\u2015])|'             # Różne typy myślników Unicode (en-dash, em-dash, itp.)
    r'(?:[.,;:!?()\[\]{}@\'\"`~&*_+=<>$#%^\\/|-])|' # Pojedyncze znaki interpunkcyjne i symbole
    r'(\s+)|'                           # Białe znaki (to jest kluczowe, żeby były osobnymi tokenami!)
    r'(?:\S)'                           # Pozostałe pojedyncze znaki niebędące białymi znakami
)

def tokenize(text):
    """
    Tokenizuje tekst wejściowy na listę tokenów i mapowanie indeksów tokenów numerycznych
    do ich oryginalnych wartości, typów i surowych reprezentacji tekstowych.
    Obsługuje specjalne tokeny dla liczb, kapitalizacji, dat, godzin i interpunkcji.
    """
    raw_entries = [] # Przechowuje (wartość, typ, surowy_string) dla tokenów <|num|>
    tokens = []

    # Normalizuj znaki Unicode i usuń typowe problemy z formatowaniem liczb
    text = unicodedata.normalize('NFKC', text)
    text = text.replace("−", "-") # Zastąp znak minus Unicode standardowym łącznikiem
    # Usuń spacje wokół przecinków w liczbach (np. "1 , 234" -> "1,234")
    text = re.sub(r'(\d)\s*,\s*(\d{3})', r'\1,\2', text)
    # Usuń spacje wokół kropek w liczbach (np. "3 . 14" -> "3.14")
    text = re.sub(r'(\d)\s*\.\s*(\d+)', r'\1.\2', text)
    
    last_idx = 0
    for match in tokenizer_pattern_re.finditer(text):
        part_str = match.group(0)
        
        if not part_str: 
            continue

        # Obsługa białych znaków jako odrębnego tokena
        if re.fullmatch(r'\s+', part_str):
            tokens.append(part_str)
            last_idx = match.end()
            continue 

        is_number_like = False
        
        # Specjalna obsługa dat, godzin, liczb szesnastkowych i ogólnych liczb
        if re.fullmatch(date_pattern_str, part_str):
            # Podziel datę na komponenty i separatory, tokenizuj komponenty jako <|num|>
            date_components_and_separators = re.split(r'([/-])', part_str)
            for comp in date_components_and_separators:
                if re.fullmatch(r'[/-]', comp):
                    tokens.append(comp)
                elif comp: 
                    try:
                        # Zawsze konwertuj na float, aby zapewnić spójność
                        val_comp = float(comp)
                        tokens.append("<|num|>")
                        raw_entries.append((val_comp, 'int_date_comp', comp))
                    except ValueError:
                        tokens.append(comp) # Powrót do domyślnego, jeśli komponent nie jest prawidłową liczbą
            is_number_like = True
        elif re.fullmatch(time_pattern_str, part_str):
            # Podziel godzinę na komponenty i separatory, tokenizuj komponenty jako <|num|>
            time_components_and_separators = re.split(r'([:])', part_str)
            for comp in time_components_and_separators:
                if comp == ':':
                    tokens.append(comp)
                elif comp:
                    try:
                        # Zawsze konwertuj na float, aby zapewnić spójność
                        val_comp = float(comp)
                        tokens.append("<|num|>")
                        raw_entries.append((val_comp, 'int_time_comp', comp))
                    except ValueError:
                        tokens.append(comp) # Powrót do domyślnego, jeśli komponent nie jest prawidłową liczbą
            is_number_like = True
        elif re.fullmatch(hex_pattern_str, part_str):
            try:
                # Konwertuj szesnastkową wartość na float
                val = float(int(part_str, 16))
                typ = 'hex'
                tokens.append("<|num|>")
                raw_entries.append((val, typ, part_str))
            except ValueError:
                tokens.append(part_str) # Powrót do domyślnego, jeśli parsowanie hex nie powiedzie się
            is_number_like = True
        elif re.fullmatch(number_pattern_str, part_str):
            try:
                clean_num_str = part_str.replace(',', '') # Usuń przecinki dla spójnego parsowania
                fv = float(clean_num_str) # Zawsze parsuj do float
                
                # Określ oryginalny typ dla metadanych, ale przechowuj float dla wartości
                if '.' in clean_num_str or 'e' in clean_num_str.lower():
                    typ = 'float'
                else:
                    typ = 'int' # Zachowaj 'int' jako type_str, jeśli pierwotnie był int
                
                val = fv # *** Przechowuj tutaj wartość float ***
                tokens.append("<|num|>")
                raw_entries.append((val, typ, part_str))
            except ValueError:
                tokens.append(part_str) # Powrót do domyślnego, jeśli parsowanie liczb nie powiedzie się
            is_number_like = True

        if is_number_like:
            last_idx = match.end() 
            continue 

        # Obsługa słów i kapitalizacji
        if re.fullmatch(r'[A-Za-z]+(?:[\'-][A-Za-z]+)*', part_str): 
            if part_str.isupper() and len(part_str) > 1:
                tokens.append("<|allcaps|>")
                tokens.append(part_str.lower())
            elif part_str[0].isupper() and (len(part_str) == 1 or part_str[1:].islower()):
                tokens.append("<|cap|>")
                tokens.append(part_str.lower())
            else:
                tokens.append(part_str)
        else:
            # Dodaj inne tokeny bezpośrednio (interpunkcję, symbole, @-tokeny itp.)
            tokens.append(part_str)
        
        last_idx = match.end() 

    # Dodaj pozostały tekst, który nie został dopasowany przez regex (powinien być rzadki przy kompleksowym regexie)
    if last_idx < len(text):
        remaining_text = text[last_idx:]
        if remaining_text:
            tokens.append(remaining_text)

    # Zbuduj number_map poprzez dopasowanie tokenów <|num|> do ich raw_entries
    number_map = {}
    raw_entry_idx = 0
    for token_idx, tok in enumerate(tokens):
        if tok == "<|num|>":
            if raw_entry_idx < len(raw_entries):
                number_map[token_idx] = raw_entries[raw_entry_idx]
                raw_entry_idx += 1
            else:
                # Powrót do domyślnego: jeśli token <|num|> istnieje bez odpowiadającego wpisu raw_entry,
                # jest to nieoczekiwany stan. Może to wskazywać na błąd w regexie lub ręczne wstawienie
                # <|num|> tam, gdzie nie powinno go być.
                number_map[token_idx] = (None, 'unknown', '<|num|>') # Placeholder
                print(f"Warning: <|num|> token at index {token_idx} has no corresponding raw_entry.")
                
    return tokens, number_map

def detokenize(tokens, number_map=None):
    """
    Detokenizuje listę tokenów z powrotem na czytelny ciąg znaków, respektując kapitalizację
    i rekonstrukcję liczb przy użyciu dostarczonego number_map.
    Obsługuje inteligentne odstępy dla interpunkcji i specjalnych tokenów.
    """
    output_parts = []
    
    cap_next_token = False
    allcaps_next_token = False

    # Zbiór tokenów, które "przywierają" do poprzedzającego tokena (brak spacji przed nimi).
    attached_to_prev = {'.', ',', ':', ';', '!', '?', ')', ']', '}', '\'', '"', '...'} 
    
    # Zbiór tokenów, które "przywierają" do następującego tokena (brak spacji po nich).
    attached_to_next = {'(', '[', '{', '\'', '"', '$'} 
    
    # Funkcja pomocnicza do sprawdzania tokenów @x...
    def is_at_token(token):
        return token.startswith('@x') and re.fullmatch(r'@[\w]+', token)

    for i, token_from_list in enumerate(tokens):
        current_word_to_append = ""

        # Obsługa znaczników kapitalizacji
        if token_from_list == '<|cap|>':
            cap_next_token = True
            continue 
        if token_from_list == '<|allcaps|>':
            allcaps_next_token = True
            continue 

        # Rekonstrukcja liczb z number_map
        if token_from_list == '<|num|>':
            if number_map and i in number_map:
                _, _, raw_number_str = number_map[i]
                current_word_to_append = raw_number_str
            else:
                current_word_to_append = '<|num|>' # Powrót do domyślnego, jeśli brak mapy lub wpisu
            cap_next_token = False # Liczby nie podlegają kapitalizacji
            allcaps_next_token = False
        else: 
            # Zastosuj kapitalizację, jeśli znaleziono znaczniki
            current_word_to_append = token_from_list
            if allcaps_next_token:
                current_word_to_append = token_from_list.upper()
                allcaps_next_token = False
            elif cap_next_token:
                current_word_to_append = token_from_list.capitalize()
                cap_next_token = False
        
        # --- LOGIKA ODSTĘPÓW ---
        if not output_parts: # Pierwszy token w całym tekście
            output_parts.append(current_word_to_append)
            continue

        # Pobierz faktyczny poprzedni token z listy `tokens` dla decyzji o odstępach
        prev_actual_token = tokens[i-1] if i > 0 else None 
        
        prev_output_part = output_parts[-1] # Ostatnia część dodana do ciągu wyjściowego

        # 1. Jeśli bieżący token to białe znaki, dodaj go bezpośrednio.
        if re.fullmatch(r'\s+', current_word_to_append):
            output_parts.append(current_word_to_append)
            continue
        
        # 2. Jeśli poprzednia część w wyjściu była białymi znakami, dodaj bieżący token bez dodatkowej spacji.
        if re.fullmatch(r'\s+', prev_output_part):
            output_parts.append(current_word_to_append)
            continue

        # 3. Jeśli bieżący token powinien przylgnąć do poprzedniego (np. interpunkcja po słowie).
        if current_word_to_append in attached_to_prev or \
           is_at_token(current_word_to_append) or \
           (current_word_to_append == '-' and prev_actual_token == '<|num|>'): # Łącznik po tokenie numerycznym (np. "10-meter")
            output_parts.append(current_word_to_append)
            continue
        
        # 4. Jeśli poprzedni token (z listy `tokens`) powinien przylgnąć do bieżącego (np. nawias otwierający).
        if prev_actual_token in attached_to_next: 
            output_parts.append(current_word_to_append)
            continue
        
        # 5. Specjalna obsługa separatorów daty/godziny ('-', ':')
        # Powinny przylegać, jeśli zarówno poprzedni, jak i następny token są liczbami.
        if (current_word_to_append == '-' or current_word_to_append == ':') and \
           prev_actual_token == '<|num|>' and \
           i + 1 < len(tokens) and tokens[i+1] == '<|num|>':
            output_parts.append(current_word_to_append)
            continue
            
        # 6. Domyślny przypadek: dodaj spację między tokenami
        output_parts.append(' ')
        output_parts.append(current_word_to_append)
            
    return ''.join(output_parts)


def summarize_tokens(tokens):
    """
    Analizuje listę tokenów w celu zliczenia częstotliwości i przypisania unikalnych identyfikatorów
    w kolejności występowania.
    Zwraca listę krotek: (ciąg_tokena, id_tokena, częstotliwość).
    """
    counts = Counter(tokens)
    token_ids = OrderedDict() # Aby zachować kolejność wstawiania
    idx_counter = 0
    for t in tokens:
        if t not in token_ids:
            token_ids[t] = idx_counter
            idx_counter +=1
            
    summary_list = []
    unique_ordered_tokens = list(token_ids.keys())

    for token_str in unique_ordered_tokens:
        summary_list.append((token_str, token_ids[token_str], counts[token_str]))
            
    return summary_list
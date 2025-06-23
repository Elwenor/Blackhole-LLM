import torch
import torch.nn as nn
import math
import struct # Do pakowania/rozpakowywania bajtów dla reprezentacji binarnej

# --- Funkcje pomocnicze do generowania i dekodowania cech liczbowych ---

def number_embedding_features(val: float, typ: str, dim: int = 128) -> torch.Tensor:
    # Obsługa wartości None, które mogą pochodzić z paddingu
    if val is None:
        return torch.full((dim,), -2.0, dtype=torch.float) # Użyj charakterystycznej wartości dla paddingu

    x = float(val)

    # Dodane: Obsługa nieskończoności i NaN na wczesnym etapie
    if math.isinf(x):
        # Dla nieskończoności, zwracamy specjalny embedding
        features = [-3.0] * dim # Użyj -3.0 jako wartości dla nieskończoności
        # Zachowaj znak w pierwszej cesze
        features[0] = math.copysign(1.0, x) if x != 0 else 0.0
        return torch.tensor(features, dtype=torch.float)
    elif math.isnan(x):
        # Dla NaN, zwracamy specjalny embedding
        return torch.full((dim,), -4.0, dtype=torch.float)

    x_abs = abs(x)
    x_sign = math.copysign(1.0, x) if x != 0 else 0.0
    
    # Log10 of absolute value, capped for bucket assignment
    # Poprawka: Jawna obsługa x_abs == 0.0, aby uniknąć log(0) i nieskończoności.
    # To jest najbezpieczniejsze podejście, aby zagwarantować, że log_abs jest zawsze skończony.
    if x_abs == 0.0:
        log_abs = -300.0 # Ustaw bardzo dużą ujemną wartość. Spowoduje to, że bucket będzie 0.
    else:
        # Używamy torch.float64, aby zapobiec underflow dla bardzo małych liczb
        log_abs = torch.log10(torch.tensor(x_abs, dtype=torch.float64)).item()
    
    # Mapujemy log_abs na 19 bucketów (np. od 10^-9 do 10^9)
    # (-9 to 9 w log10 skali, offset +9 do zakresu 0-18)
    bucket = min(max(int(math.floor(log_abs + 9)), 0), 18) # 19 bucketów total
    
    # Binary encoding of the float64 value (64 bits)
    # Konwertujemy float na 8 bajtów (64 bity) w formacie double-precision
    binary = struct.pack('>d', x) # '>d' oznacza big-endian double
    bits = ''.join(format(byte, '08b') for byte in binary) # Konwertujemy bajty na ciąg bitów
    # Poprawka: -1/1 encoding dla stabilności treningu.
    binary_features = [1.0 if b == '1' else -1.0 for b in bits] 
    
    # Semantic features to help model generalize
    semantic_features = [
        x_sign,                                     # znak (+1/-1/0)
        1.0 if x == 0 else -1.0,                    # czy jest zerem
        1.0 if typ == 'int' else -1.0,              # czy jest int
        1.0 if typ == 'float' else -1.0,            # czy jest float
        1.0 if typ == 'hex' else -1.0,              # czy jest hex
        1.0 if typ == 'int_date_comp' else -1.0,    # czy jest komponentem daty
        1.0 if typ == 'int_time_comp' else -1.0,    # czy jest komponentem czasu
        math.tanh(x / 1000.0),                      # tanh(x) dla małych/średnich wartości (ograniczony zakres)
        math.tanh(log_abs / 20.0),                  # tanh(log_abs) dla rzędu wielkości (również ograniczony)
    ]
    
    # Bucket one-hot z kodowaniem -1/1
    bucket_features = [-1.0] * 19
    bucket_features[bucket] = 1.0
    
    # Połącz wszystkie cechy
    features = semantic_features + bucket_features + binary_features
    
    # Wypełnij lub przytnij do żądanego wymiaru
    # Upewnij się, że dim jest wystarczająco duże, aby pomieścić wszystkie cechy (9 semantycznych + 19 bucketów + 64 binarne = 92)
    min_required_dim = 9 + 19 + 64 # Suma wszystkich generowanych cech
    if dim < min_required_dim:
        print(f"Warning: `dim` ({dim}) is less than the minimum required ({min_required_dim}). Some features will be truncated.")
        features = features[:dim]
    elif len(features) < dim:
        features += [-2.0] * (dim - len(features)) # Użyj -2.0 jako wartości paddingu
    
    return torch.tensor(features, dtype=torch.float) # Output tensor can still be float32, but input to log10 must be float64

def decode_number_from_features(features: torch.Tensor) -> float:
    # KLUCZOWA ZMIANA: Upewnij się, że 'features' jest tensorem PyTorcha
    if not isinstance(features, torch.Tensor):
        # Zakładamy, że jeśli nie jest tensorem, to jest numpy.ndarray
        # Konwertujemy na tensor PyTorcha. Domyślnie na CPU.
        features = torch.from_numpy(features) 
        # Opcjonalnie, jeśli chcesz, aby tensor był na tym samym urządzeniu co model podczas ewaluacji,
        # możesz przekazać 'device' do tej funkcji i użyć .to(device).
        # Na razie pozostawienie na CPU jest bezpieczne, ponieważ operacje dekodowania nie są intensywne GPU.

    # Sprawdź, czy to cecha paddingu
    if torch.all(features == -2.0):
        return float('nan') # Zwróć NaN dla paddingu
    
    # Dodane: Obsługa nieskończoności i NaN z embeddingu
    if torch.all(features == -3.0): # Zakładamy, że -3.0 jest dla nieskończoności
        # Spróbuj odzyskać znak, jeśli został zakodowany
        if features[0].item() > 0.5:
            return float('inf')
        elif features[0].item() < -0.5:
            return float('-inf')
        else: # Domyślnie pozytywna nieskończoność, jeśli znak jest niejednoznaczny
            return float('inf')
    elif torch.all(features == -4.0): # Zakładamy, że -4.0 jest dla NaN
        return float('nan')

    # Upewniamy się, że tensor jest na CPU i nie śledzi gradientów, aby uniknąć błędów
    # Ta linia jest nadal dobra, ale teraz 'features' jest już tensorem
    features_processed = features.clone().detach().cpu() 

    # Poprawka: Upewnij się, że minimalna długość jest zawsze 92 dla pełnego dekodowania.
    min_len_for_full_decode = 9 + 19 + 64 

    # --- ULEPSZONE DEKODOWANIE BINARNE ---
    # Spróbuj pełnego dekodowania binarnego, jeśli cechy są wystarczająco długie
    if len(features_processed) >= min_len_for_full_decode:
        # Pobieramy segment cech binarnych
        binary_features_segment = features_processed[9 + 19 : 9 + 19 + 64] 

        try:
            # KLUCZOWA ZMIANA: Robust conversion from [-1, 1] to binary bits [0, 1].
            # Przesuwamy zakres z [-1, 1] do [0, 2], dzielimy przez 2 do [0, 1].
            # Następnie zaokrąglamy do najbliższej liczby całkowitej (0 lub 1).
            rounded_bits = torch.round((binary_features_segment + 1.0) / 2.0).long()

            # Konwertujemy zaokrąglone bity na string '0's i '1's
            binary_str = ''.join(str(int(b.item())) for b in rounded_bits)
            
            # Zapewniamy, że string binarny ma dokładnie 64 bity
            if len(binary_str) < 64:
                binary_str += '0' * (64 - len(binary_str)) # Dopełniamy zerami, jeśli za krótki
            elif len(binary_str) > 64: 
                binary_str = binary_str[:64] # Przycinamy, jeśli za długi (nie powinno się dziać z stałym wymiarem)

            binary_bytes = bytearray()
            # Konwertujemy 8-bitowe chunki stringu binarnego na bajty
            for i in range(0, len(binary_str), 8):
                if i + 8 <= len(binary_str): # Upewnij się, że mamy pełny bajt
                    byte = int(binary_str[i : i + 8], 2)
                    binary_bytes.append(byte)
            
            # Rozpakowujemy 8 bajtów (64 bity) do liczby zmiennoprzecinkowej podwójnej precyzji
            if len(binary_bytes) == 8: 
                return struct.unpack('>d', bytes(binary_bytes))[0]
        except (ValueError, struct.error, IndexError) as e:
            # print(f"Binary decoding error: {e}") # Odkomentuj do debugowania, jeśli potrzebne
            pass # Fallback do przybliżonego dekodowania, jeśli binarne zawiedzie
    
    # --- FALLBACK: Dekodowanie przybliżone (jeśli dekodowanie binarne zawiedzie) ---
    # Jest mniej precyzyjne, ale może dać rozsądny wynik.
    
    # Sprawdź, czy cechy są wystarczająco długie dla podstawowych cech semantycznych (minimum 9)
    if len(features_processed) < 9: 
        # print(f"[WARN] Cechy za krótkie do podstawowego dekodowania: {len(features_processed)}")
        return 0.0 # Lub float('nan') jako fallback

    x_sign_feat = features_processed[0].item()
    is_zero_feat = features_processed[1].item() 
    
    # Poprawka: Jeśli cecha 'is_zero' jest silnie pozytywna (np. > 0.8, reprezentując '1.0'), zwróć 0.0.
    # To jest priorytet, ponieważ zero jest dokładnie reprezentowane.
    if is_zero_feat > 0.8: # Ustawiamy próg np. 0.8 dla '1.0'
        return 0.0

    # Próba odwrócenia tanh(x / 1000.0)
    tanh_x_scaled = features_processed[7].item() 
    # Ogranicz wartości, aby uniknąć problemów z domenami atanh, z niewielkim buforem
    if tanh_x_scaled >= 1.0:
        tanh_x_scaled = 1.0 - 1e-7
    if tanh_x_scaled <= -1.0:
        tanh_x_scaled = -1.0 + 1e-7

    try:
        x_approx_from_tanh = math.atanh(tanh_x_scaled) * 1000.0
    except ValueError: 
        x_approx_from_tanh = 0.0 

    # Użyj log_abs (features[8]) i bucketów, aby uściślić predykcję, zwłaszcza dla dużych/małych liczb
    tanh_log_abs = features_processed[8].item() 
    if tanh_log_abs >= 1.0:
        tanh_log_abs = 1.0 - 1e-7
    if tanh_log_abs <= -1.0:
        tanh_log_abs = -1.0 + 1e-7

    log_abs_approx_from_tanh = 0.0
    try:
        log_abs_approx_from_tanh = math.atanh(tanh_log_abs) * 20.0
    except ValueError:
        pass # Ignoruj błędy atanh
        
    # Znajdź najbardziej aktywny bucket
    pred_bucket_idx = -1
    if len(features_processed) >= 9 + 19: # Upewnij się, że są buckety
        bucket_features = features_processed[9 : 9 + 19]
        # Użyj torch.max, aby znaleźć indeks i wartość najwyższego elementu
        pred_bucket_val, pred_bucket_idx_tensor = bucket_features.max(dim=0)
        if pred_bucket_val.item() > 0.5: # Upewnij się, że bucket jest z pewnością aktywowany
            pred_bucket_idx = pred_bucket_idx_tensor.item()
    
    log_abs_from_bucket = 0.0
    if pred_bucket_idx != -1:
        log_abs_from_bucket = pred_bucket_idx - 9.0 

    # Poprawka: Bardziej stabilne połączenie informacji z tanh(log_abs) i bucketu.
    # Dajemy większą wagę bucketowi, jeśli jest silnie aktywowany, ponieważ daje on informację o rzędzie wielkości.
    # W przeciwnym razie polegamy na tanh(log_abs).
    
    final_log_abs = log_abs_approx_from_tanh # Domyślnie użyj estymaty z tanh
    
    if pred_bucket_idx != -1 and pred_bucket_val.item() > 0.5:
        # Jeśli bucket jest silny, uśrednij go z log_abs z tanh, dając przewagę bucketowi
        final_log_abs = (log_abs_approx_from_tanh + log_abs_from_bucket * 2) / 3 # Daj 2x wagę bucketowi
    
    # Rekonstrukcja wartości bezwzględnej
    abs_reconstructed_val = 0.0
    if abs(final_log_abs) > 1e-6: # Jeśli logarytm jest sensowny
        abs_reconstructed_val = 10 ** final_log_abs
    
    # Poprawka: Ostateczne połączenie:
    # 1. Jeśli z binarnych cech nie udało się odtworzyć, a cecha "jest zero" jest aktywowana, to 0.0. (Już jest)
    # 2. Jeśli mamy silne estymaty z log_abs (czyli jest to duża/mała liczba), użyj jej.
    # 3. Jeśli nie, użyj estymaty z tanh(x).
    
    final_val_candidate = x_approx_from_tanh # Domyślna estymata z tanh(x)
    
    # Jeśli abs_reconstructed_val z log_abs jest znacząco różna od 0
    if abs(abs_reconstructed_val) > 1e-5: # Jeśli to jest znacząca wartość
        # Jeśli tanh(x) daje bardzo małą wartość, ale log_abs daje dużą, zaufaj log_abs
        if abs(x_approx_from_tanh) < 1e-3 and abs(abs_reconstructed_val) > 1e-3:
            final_val_candidate = abs_reconstructed_val
        else: # W przeciwnym razie uśrednij, lub wybierz bardziej pewną wartość (to jest heurystyczne)
            # Można tu zastosować bardziej zaawansowane łączenie, np. w oparciu o odchylenie od środka tanh.
            # Na razie prosta średnia, która jest bezpieczniejsza niż agresywne wybieranie.
            final_val_candidate = (x_approx_from_tanh + abs_reconstructed_val) / 2.0
    
    # Poprawka: Zastosuj znak na samym końcu.
    # Normalizujemy cechę znaku do -1, 0 lub 1
    final_sign = 0.0
    if x_sign_feat > 0.5: # Bliskie 1.0
        final_sign = 1.0
    elif x_sign_feat < -0.5: # Bliskie -1.0
        final_sign = -1.0
    # W przeciwnym razie pozostaje 0.0 (dla prawdziwego zera, które już zostało obsłużone, ale też dla zaszumionego sygnału)

    if final_sign == 0.0:
        return 0.0 # Jeśli znak jest 0, to liczba jest 0.
    else:
        return abs(final_val_candidate) * final_sign


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

class NumberEmbedding(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 128):
        super().__init__()
        # Prosta sieć liniowa do transformacji cech numerycznych
        # Może być bardziej złożona (np. z warstwami non-liniowymi)
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU() # Dodajemy aktywację dla nieliniowości

    def forward(self, numeric_features: torch.Tensor) -> torch.Tensor:
        # numeric_features powinny mieć kształt (batch_size, seq_len, input_dim)
        return self.relu(self.fc(numeric_features))


# --- Funkcja prepare_inputs (NOWA) ---

def prepare_inputs(tokens: list[str], number_map: dict, dim: int = 128) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Przygotowuje tokeny i cechy numeryczne do wejścia do modeli embeddingowych.

    Args:
        tokens: Lista tokenów (stringów) zwróconych przez tokenizer.
        number_map: Słownik mapujący indeks tokena <|num|> na jego oryginalną wartość
                    i typ (zwrócony przez tokenizer).
        dim: Oczekiwany wymiar cech numerycznych generowanych przez number_embedding_features.

    Returns:
        Tuple zawierający:
        - token_ids (torch.Tensor): Tensor identyfikatorów tokenów dla TokenEmbedding.
                                    Kształt: [1, długość_sekwencji].
        - numeric_features_tensor (torch.Tensor): Tensor cech numerycznych dla NumberEmbedding.
                                               Kształt: [1, długość_sekwencji, dim].
        - vocab (dict): Słownik mapujący tokeny na ich identyfikatory.
    """
    
    # 1. Budowanie słownika (vocab) i mapowanie tokenów na ID
    vocab = {
        "<|unk|>": 0, # Token dla nieznanych tokenów
        "<|pad|>": 1, # Token dla paddingu
        "<|num|>": 2, # Specjalny token dla liczb
        "<|cap|>": 3, # Special token for capitalization (from tokenizer)
        "<|space|>": 4 # Special token for spaces (from tokenizer)
    }
    current_vocab_id = len(vocab)
    
    # Dodaj wszystkie unikalne tokeny do słownika
    for token in tokens:
        if token not in vocab:
            vocab[token] = current_vocab_id
            current_vocab_id += 1

    # 2. Generowanie token_ids
    token_ids_list = [vocab.get(token, vocab["<|unk|>"]) for token in tokens]
    token_ids = torch.tensor([token_ids_list], dtype=torch.long) # Kształt [1, L]

    # 3. Generowanie tensora cech numerycznych
    seq_len = len(tokens)
    # Inicjalizujemy tensor cech numerycznych wypełniony wartościami paddingu (-2.0)
    numeric_features_tensor = torch.full((1, seq_len, dim), -2.0, dtype=torch.float)

    # Wypełnij tensor cechami dla tokenów liczbowych
    for token_idx, (val, typ, raw) in number_map.items():
        # Upewnij się, że token_idx jest w zakresie sekwencji
        if token_idx < seq_len:
            features = number_embedding_features(val, typ, dim=dim)
            numeric_features_tensor[0, token_idx] = features

    return token_ids, numeric_features_tensor, vocab
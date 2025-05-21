import torch
import torch.nn as nn
import math
import struct # Do pakowania/rozpakowywania bajtów dla reprezentacji binarnej

# Nowa, zaawansowana funkcja generowania cech numerycznych
def number_embedding_features(val: float, typ: str, dim: int = 128) -> torch.Tensor:
    # Obsługa wartości None, które mogą pochodzić z paddingu
    if val is None:
        return torch.full((dim,), -2.0, dtype=torch.float) # Użyj charakterystycznej wartości dla paddingu

    x = float(val)
    x_abs = abs(x)
    x_sign = math.copysign(1.0, x) if x != 0 else 0.0
    
    # Log10 of absolute value, capped for bucket assignment
    # Dodajemy 1e-12 aby uniknąć log(0) i zapewnić stabilność
    log_abs = math.log10(x_abs + 1e-12) if x_abs > 0 else -12.0
    # Mapujemy log_abs na 19 bucketów (np. od 10^-9 do 10^9)
    # (-9 to 9 w log10 skali)
    bucket = min(max(int(math.floor(log_abs + 9)), 0), 18) # 19 bucketów total
    
    # Binary encoding of the float64 value (64 bits)
    # Konwertujemy float na 8 bajtów (64 bity) w formacie double-precision
    binary = struct.pack('>d', x) # '>d' oznacza big-endian double
    bits = ''.join(format(byte, '08b') for byte in binary) # Konwertujemy bajty na ciąg bitów
    binary_features = [1.0 if b == '1' else -1.0 for b in bits] # -1/1 encoding dla stabilności treningu
    
    # Semantic features to help model generalize
    semantic_features = [
        x_sign,                                  # znak (+1/-1/0)
        1.0 if x == 0 else -1.0,                 # czy jest zerem
        1.0 if typ == 'int' else -1.0,           # czy jest int
        1.0 if typ == 'float' else -1.0,         # czy jest float
        1.0 if typ == 'hex' else -1.0,           # czy jest hex
        1.0 if typ == 'int_date_comp' else -1.0, # czy jest komponentem daty
        1.0 if typ == 'int_time_comp' else -1.0, # czy jest komponentem czasu
        math.tanh(x / 1000.0),                   # tanh(x) dla małych/średnich wartości (ograniczony zakres)
        math.tanh(log_abs / 20.0),               # tanh(log_abs) dla rzędu wielkości (również ograniczony)
    ]
    
    # Bucket one-hot z kodowaniem -1/1
    bucket_features = [-1.0] * 19
    bucket_features[bucket] = 1.0
    
    # Połącz wszystkie cechy
    features = semantic_features + bucket_features + binary_features
    
    # Wypełnij lub przytnij do żądanego wymiaru
    if len(features) < dim:
        features += [-2.0] * (dim - len(features)) # Użyj -2.0 jako wartości paddingu
    else:
        features = features[:dim]
    
    return torch.tensor(features, dtype=torch.float)

# Funkcja dekodująca cechy numeryczne z powrotem do liczby
def decode_number_from_features(features: torch.Tensor) -> float:
    # Sprawdź, czy to cecha paddingu
    if torch.all(features == -2.0):
        return float('nan') # Zwróć NaN dla paddingu

    # Minimalna długość potrzebna do pełnego dekodowania z binarnych cech (semantic + bucket + binary)
    # 9 semantic + 19 bucket + 64 binary = 92
    min_len_for_full_decode = 9 + 19 + 64 

    # Upewniamy się, że tensor jest na CPU i nie śledzi gradientów, aby uniknąć błędów
    features_processed = features.clone().detach().cpu() 

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
    
    # Jeśli cecha 'is_zero' jest silnie pozytywna (np. > 0.8, reprezentując '1.0'), zwróć 0.0
    if is_zero_feat > 0.8: # Ustawiamy próg np. 0.8 dla '1.0'
        return 0.0

    # Próba odwrócenia tanh(x / 1000.0)
    tanh_x_scaled = features_processed[7].item() 
    # Ogranicz wartości, aby uniknąć problemów z domeną atanh, z niewielkim buforem
    if tanh_x_scaled >= 1.0:
        tanh_x_scaled = 1.0 - 1e-7
    if tanh_x_scaled <= -1.0:
        tanh_x_scaled = -1.0 + 1e-7

    try:
        x_approx = math.atanh(tanh_x_scaled) * 1000.0
    except ValueError: 
        x_approx = 0.0 

    # Użyj log_abs (features[8]) i bucketów, aby uściślić predykcję, zwłaszcza dla dużych/małych liczb
    tanh_log_abs = features_processed[8].item() 
    if tanh_log_abs >= 1.0:
        tanh_log_abs = 1.0 - 1e-7
    if tanh_log_abs <= -1.0:
        tanh_log_abs = -1.0 + 1e-7

    try:
        log_abs_approx = math.atanh(tanh_log_abs) * 20.0
        
        # Znajdź najbardziej aktywny bucket
        bucket_features = features_processed[9 : 9 + 19]
        # Użyj torch.max, aby znaleźć indeks i wartość najwyższego elementu
        pred_bucket_val, pred_bucket_idx = bucket_features.max(dim=0)
        
        if pred_bucket_val.item() > 0.5: # Upewnij się, że bucket jest z pewnością aktywowany
            # Oszacuj log_abs na podstawie indeksu bucketu
            log_abs_from_bucket = pred_bucket_idx.item() - 9.0 
            
            # Połącz informacje z tanh(log_abs) i bucketu
            log_abs_final = (log_abs_approx + log_abs_from_bucket) / 2.0 
            
            # Oblicz wartość z logarytmu, stosując znak
            # Jeśli x_approx jest bardzo mały, ale log_abs_final wskazuje na dużą liczbę, zaufaj log_abs
            if abs(log_abs_final) > 1e-6:
                x_from_log = (10 ** log_abs_final) * x_sign_feat
                if abs(x_from_log) > abs(x_approx) or abs(x_approx) < 1e-3:
                    x_approx = x_from_log

    except ValueError:
        pass # Ignoruj błędy atanh

    # Zastosuj cechę znaku dla końcowego wyniku. Jeśli sign_feat jest 0 (dla prawdziwego zera), nie modyfikuj x_approx.
    return x_approx * x_sign_feat if x_sign_feat != 0 else x_approx

# Klasa NumberEmbedding - Teraz prostsza, ponieważ number_embedding_features już robi większość pracy
class NumberEmbedding(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 128): # output_dim może być równe input_dim, jeśli chcemy mapować 1:1
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Pojedyncza warstwa liniowa do mapowania cech wejściowych do przestrzeni osadzeń
        # Możesz dodać ReLU, jeśli chcesz nieliniowość po projekcji
        self.projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Pamiętaj, że features mogą zawierać padding (-2.0), który powinien być rozpoznany przez sieć
        return self.projection(features)

# Klasa TokenEmbedding - Bez zmian
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, token_ids):
        return self.embedding(token_ids)
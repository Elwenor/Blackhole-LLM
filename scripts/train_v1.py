import sys
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from num2words import num2words
import matplotlib.pyplot as plt


# Safe __file__ usage with proper error handling
try:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    # Fallback if __file__ is not defined (e.g., in interactive environments)
    base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, base_dir)

# Set seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Import z Twoich modułów blackhole
from blackhole.tokenizer import *
from blackhole.embedding import *
import blackhole.embedding

print(f"DEBUG: Załadowano embedding.py z: {blackhole.embedding.__file__}")

# --- DEFINICJA MODELU ---
class ImprovedCrossEmbeddingModel(nn.Module):
    """
    Model łączący osadzenia tokenów i liczb za pomocą TransformerEncoder.
    Zaprojektowany do zadań obejmujących tekst i liczby (np. pytania arytmetyczne).
    """
    def __init__(self, vocab_size, token_dim=128, num_dim=128, hidden=256, num_layers=3, dropout=0.2, feature_dim=128):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=token_dim)
        # NumberEmbedding teraz przyjmuje 'feature_dim' jako input_dim i 'num_dim' jako output_dim
        self.num_embedding = NumberEmbedding(input_dim=feature_dim, output_dim=num_dim) 
        
        # Projekcje do wyrównania wymiarów, jeśli token_dim != num_dim
        self.token_to_common_dim = nn.Linear(token_dim, hidden) # mapuje token_emb do hidden
        self.num_to_common_dim = nn.Linear(num_dim, hidden)      # mapuje num_emb do hidden
        
        # Warstwy Transformera (d_model powinno być hidden - wspólny wymiar)
        # Zwiększ liczbę warstw i/lub wymiar ukryty oraz feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, # Wspólny wymiar dla Transformera
            nhead=8,
            dim_feedforward=hidden * 4, # Zwiększ FeedForward - więcej "myślenia" na token
            dropout=dropout, # Pozostawiamy na 0.1, ale można eksperymentować
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=5) # Zwiększ do 5 warstw
        
        # Głowice wyjściowe
        self.token_head = nn.Linear(hidden, vocab_size) 
        
        # Głowica numeryczna z common_dim do feature_dim.
        # Dodajemy warstwę aktywacji Tanh, aby wymusić wyjście w zakresie (-1, 1),
        # co jest zgodne z Twoim kodowaniem cech numerycznych.
        self.num_head = nn.Sequential(
            nn.Linear(hidden, feature_dim),
            nn.Tanh() 
        )
        
    def forward(self, token_ids, numeric_features, attention_mask=None, return_emb=False):
        """
        Przebieg do przodu przez model.
        
        Args:
            token_ids: Tensor ID tokenów [batch_size, seq_len]
            numeric_features: Tensor cech numerycznych [batch_size, seq_len, feature_dim]
                                     (wyjście number_embedding_features dla każdego tokenu numerycznego)
            attention_mask: Maska uwagi [batch_size, seq_len] (1 dla prawdziwego tokenu, 0 dla paddingu)
            return_emb: Czy zwrócić pośrednie osadzenia (dla straty wyrównania)
            
        Returns:
            Krotka wyjść (logits tokenów, wyjście numeryczne, pośrednie osadzenia jeśli return_emb=True)
        """
        # Osadzenia tokenów
        token_emb = self.token_embedding(token_ids) # [batch_size, seq_len, token_dim]
        
        # Osadzenia numeryczne (przy użyciu modułu NumberEmbedding)
        num_emb = self.num_embedding(numeric_features) # [batch_size, seq_len, num_dim]
        
        # Projekcje osadzeń do wspólnej przestrzeni (hidden_dim)
        token_emb_proj = self.token_to_common_dim(token_emb) # [batch_size, seq_len, hidden]
        num_emb_proj = self.num_to_common_dim(num_emb)       # [batch_size, seq_len, hidden]
        
        # Łączymy osadzenia przed Transformerem. 
        combined_emb = token_emb_proj + num_emb_proj # [batch_size, seq_len, hidden]
        
        # Tworzymy maskę paddingu dla transformera (Transformer oczekuje True dla zamaskowanych elementów)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0) # [batch_size, seq_len]
        else:
            padding_mask = None
        
        # Przechodzimy przez transformer
        transformer_out = self.transformer(combined_emb, src_key_padding_mask=padding_mask)
        
        # Głowica klasyfikacji tokenów
        token_logits = self.token_head(transformer_out) # [batch_size, seq_len, vocab_size]
        
        # Głowica predykcji numerycznej
        num_feature_out = self.num_head(transformer_out) # [batch_size, seq_len, feature_dim]
        
        if return_emb:
            return token_logits, token_emb_proj, num_emb_proj, num_feature_out
        else:
            return token_logits, num_feature_out


# --- PRZYGOTOWANIE DANYCH ---
def prepare_batch(batch, vocab, feature_embedding_dim=128):
    """
    Przygotowuje partię przykładów do treningu lub ewaluacji modelu.
    Ta wersja poprawnie wykorzystuje `number_map` z `tokenize` do generowania cech
    przy użyciu `number_embedding_features`.
    
    Args:
        batch: Lista krotek (text, target_numeric_str, a_info, b_info, c_info)
               gdzie c_info = (value, type, raw_string) dla numerycznego celu.
        vocab: Słownik mapujący tokeny na indeksy
        feature_embedding_dim: Oczekiwany wymiar wyjścia number_embedding_features.
                               Musi pasować do `input_dim` w `NumberEmbedding`.
        
    Returns:
        Krotka tensorów:
        - token_ids: [batch_size, max_seq_len]
        - numeric_features: [batch_size, max_seq_len, feature_embedding_dim] (zainicjalizowany wartością paddingu)
        - attention_masks: [batch_size, max_seq_len] (1 dla prawdziwego tokenu, 0 dla paddingu)
        - numeric_targets_features: [batch_size, feature_embedding_dim] (wektor cech dla prawdziwej odpowiedzi)
        - original_mappings: Lista słowników `number_map` dla każdego przykładu
    """
    token_ids_list, numeric_features_list, attention_masks_list, numeric_targets_features_list = [], [], [], []
    original_mappings_list = [] 

    pad_token_id = vocab.get('<|pad|>', 0)
    
    for text_input, target_str_info, a_info, b_info, c_info in batch:
        tokens, number_map = tokenize(text_input) 
        
        ids = [vocab.get(tok, vocab.get('<|unk|>', 0)) for tok in tokens]
        
        # Inicjujemy cechy numeryczne dla tej sekwencji
        # Wartość paddingu dla cech numerycznych powinna być taka sama jak w number_embedding_features (-2.0)
        seq_feats = torch.full((len(ids), feature_embedding_dim), -2.0, dtype=torch.float)
        
        # Wypełniamy cechy numeryczne na podstawie number_map
        for token_idx, (val, typ, raw) in number_map.items():
            if token_idx < len(ids): 
                seq_feats[token_idx] = number_embedding_features(val, typ, dim=feature_embedding_dim)
        
        attn = [1] * len(ids)
        
        token_ids_list.append(ids)
        numeric_features_list.append(seq_feats.tolist()) 
        attention_masks_list.append(attn)
        
        # Cel numeryczny to teraz WEKTOR CECH
        numeric_targets_features_list.append(
            number_embedding_features(float(target_str_info[0]), target_str_info[1], dim=feature_embedding_dim).tolist()
        )
        original_mappings_list.append(number_map)

    max_len = max(len(x) for x in token_ids_list)

    padded_token_ids = []
    padded_numeric_features = []
    padded_attention_masks = []
    
    # Wartość paddingu dla cech numerycznych
    padded_feat_row = [-2.0] * feature_embedding_dim 

    for ids, feats, mask in zip(token_ids_list, numeric_features_list, attention_masks_list):
        padded_ids = ids + [pad_token_id] * (max_len - len(ids))
        padded_token_ids.append(padded_ids)
        
        padded_feats = feats + [padded_feat_row] * (max_len - len(feats))
        padded_numeric_features.append(padded_feats)
        
        padded_mask = mask + [0] * (max_len - len(mask))
        padded_attention_masks.append(padded_mask)
    
    token_ids_tensor = torch.tensor(padded_token_ids, dtype=torch.long)
    numeric_features_tensor = torch.tensor(padded_numeric_features, dtype=torch.float32)
    attention_masks_tensor = torch.tensor(padded_attention_masks, dtype=torch.long)
    numeric_targets_features_tensor = torch.tensor(numeric_targets_features_list, dtype=torch.float32) 

    return token_ids_tensor, numeric_features_tensor, attention_masks_tensor, numeric_targets_features_tensor, original_mappings_list

def generate_improved_examples(n=1000, max_num=10000):
    """
    Generuje przykłady do treningu i ewaluacji.
    """
    templates = [
        "{a} + {b} =", "What is {a} plus {b}?", "Add {a} and {b}",
        "Compute sum of {a} and {b}", "The sum of {a} and {b} is",
        "{a} added to {b} equals", "Calculate {a} + {b}", "If you add {a} and {b}, you get",
        "What do you get when you add {a} and {b}?", "{a} plus {b} equals"
    ]

    def fmt(x, style=None):
        """Formatuje liczbę jako słowa lub cyfry w zależności od stylu"""
        if style is None:
            style = random.choice(['words', 'digits', 'digits'])
        
        raw_val = str(x)
        if style == 'words':
            try:
                text_val = num2words(x, lang='en')
            except NotImplementedError: 
                text_val = str(x) 
            return (text_val, 'words', raw_val)
        else:
            return (str(x), 'digits', raw_val)

    samples = []
    for _ in range(n):
        a, b = (random.randint(0, 100), random.randint(0, 100)) if random.random() < 0.7 else \
               (random.randint(0, max_num), random.randint(0, max_num))
        c = a + b
        
        a_s, a_type, a_raw = fmt(a, random.choice(['digits', 'digits', 'words']))
        b_s, b_type, b_raw = fmt(b, random.choice(['digits', 'digits', 'words']))
        
        c_raw_str = str(c) 
        # Zmieniamy typ na 'int' dla liczby całkowitej, bo to jest suma
        c_info = (c, 'int', c_raw_str) # (wartość, typ, raw_string)
        
        inp = random.choice(templates).format(a=a_s, b=b_s)
        samples.append((inp, c_info, (a, a_type, a_raw), (b, b_type, b_raw), c_info))
    
    return samples

def build_vocab_from_examples(examples):
    """
    Buduje słownik z przykładów.
    """
    tokens_all = []
    for inp, out_info, *_ in examples:
        toks_inp, _ = tokenize(inp)
        toks_out, _ = tokenize(out_info[2]) # Używamy raw_string reprezentacji liczby docelowej
        
        tokens_all.extend(toks_inp)
        tokens_all.extend(toks_out)
    
    special_tokens = {'<|pad|>', '<|unk|>', '<|bos|>', '<|eos|>', '<|cap|>', '<|allcaps|>', '<|num|>', '<|space|>'}
    unique_tokens = set(tokens_all) | special_tokens
    
    vocab = {tok: i for i, tok in enumerate(sorted(unique_tokens))}
    return vocab

# --- FUNKCJE STRAT ---
def focal_loss(logits, targets, alpha=0.25, gamma=2.0, ignore_index=0):
    """Oblicza focal loss dla klasyfikacji."""
    ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()

def cosine_similarity_loss(a, b, attention_mask=None):
    """
    Oblicza stratę podobieństwa kosinusowego między przestrzeniami osadzeń,
    z uwzględnieniem maski uwagi.
    
    Args:
        a, b: Tensory osadzeń do porównania [batch_size, seq_len, dim]
        attention_mask: Maska uwagi [batch_size, seq_len]
        
    Returns:
        Uśredniona strata podobieństwa kosinusowego.
    """
    if attention_mask is not None:
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(a)
        a = a * mask_expanded
        b = b * mask_expanded
    
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)

    a_norm[a_norm.isnan()] = 0.0
    b_norm[b_norm.isnan()] = 0.0

    sim = (a_norm * b_norm).sum(dim=-1) 
    
    if attention_mask is not None:
        sim = sim * attention_mask
        num_elements = attention_mask.sum().float()
    else:
        num_elements = torch.numel(sim) 
    
    if num_elements > 0:
        return (1 - sim).sum() / num_elements
    else:
        return torch.tensor(0.0, device=a.device)


def mse_loss_for_features(pred_features, target_features):
    """
    Oblicza błąd średniokwadratowy (MSE) dla wektorów cech numerycznych.
    Nie stosujemy tutaj maski, ponieważ `pred_features` i `target_features`
    są już zredukowane do [batch_size, feature_dim].
    """
    diff = pred_features - target_features
    sq_diff = diff ** 2
    
    return sq_diff.mean()


# --- LOGIKA TRENINGU I EWALUACJI ---
def train_step(model, optimizer, scheduler, batch, vocab, device, clip_grad=1.0):
    """Wykonuje jeden krok treningowy."""
    model.train()
    
    tok_ids, feats, attention_mask, num_targets_features, _ = prepare_batch(batch, vocab)
    tok_ids, feats, attention_mask, num_targets_features = [x.to(device) for x in (tok_ids, feats, attention_mask, num_targets_features)]
    
    optimizer.zero_grad()
    logits, token_emb_proj, num_emb_proj, num_feature_out = model(tok_ids, feats, attention_mask, return_emb=True)
    
    pad_id = vocab.get('<|pad|>', 0)
    
    ce_loss = focal_loss(logits.view(-1, logits.size(-1)), tok_ids.view(-1), ignore_index=pad_id)
    
    align_loss = cosine_similarity_loss(token_emb_proj, num_emb_proj, attention_mask=attention_mask)

    last_valid_token_indices = attention_mask.sum(dim=1) - 1 

    predicted_num_features_batch = num_feature_out[torch.arange(num_feature_out.size(0)), last_valid_token_indices] 

    # Zwiększ wagę dla numeric_loss, aby model bardziej koncentrował się na precyzji
    # Zaczniemy od 0.5, co jest znacznym zwiększeniem. Możesz eksperymentować z tą wartością (np. 0.2, 1.0, 5.0)
    numeric_loss = mse_loss_for_features(predicted_num_features_batch, num_targets_features)
    loss = ce_loss + 0.15 * align_loss + 0.5 * numeric_loss 

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    return {
        'ce': ce_loss.item(), 
        'align': align_loss.item(), 
        'num_loss': numeric_loss.item(), 
        'total': loss.item(), 
        'num_values': num_targets_features.shape[0] 
    }

def evaluate(model, examples, vocab, device):
    """Ocenia model na przykładach."""
    model.eval()
    predictions_decoded_values = [] 
    batch_size = 32 
    
    # Dodane: Lista do przechowywania wyników debugowania
    debug_outputs = [] # DODANE

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        tok_ids, feats, attention_mask, _, _ = prepare_batch(batch, vocab) 
        
        with torch.no_grad():
            logits, num_feature_out = model(tok_ids.to(device), feats.to(device), attention_mask.to(device))
        
        last_valid_token_indices = attention_mask.sum(dim=1).cpu() - 1 

        predicted_num_features_batch = num_feature_out[torch.arange(num_feature_out.size(0)), last_valid_token_indices].cpu() 
        
        for j, target_example in enumerate(batch):
            decoded_val = decode_number_from_features(predicted_num_features_batch[j])
            
            # DODANE: Pobieranie oryginalnych danych i targetu
            original_text_input = target_example[0] # To jest oryginalne pytanie
            true_val_info = target_example[1]       # To jest krotka (wartość, typ, raw_string)
            true_val = float(true_val_info[0])
            true_raw_str = true_val_info[2]
            # KONIEC DODANYCH

            if not math.isnan(decoded_val):
                final_pred_str = str(round(decoded_val)) 
            else:
                final_pred_str = "NaN" 

            predictions_decoded_values.append(final_pred_str) 

            # DODANE: Zapisywanie informacji do debugowania
            debug_outputs.append({
                'input': original_text_input,
                'target_value': true_val,
                'target_str': true_raw_str,
                'predicted_str': final_pred_str,
                'predicted_val': decoded_val, # Dodajemy też niedokładnie zaokrągloną wartość
                'is_correct': (abs(round(decoded_val) - round(true_val)) < 0.5) if not math.isnan(decoded_val) else False
            })
            # KONIEC DODANYCH
    
    # DODANE: Wyświetlanie przykładów debugowania po zakończeniu ewaluacji
    print("\n--- Przykłady walidacyjne i predykcje (debug) ---")
    # Wyświetlmy tylko kilka przykładów, aby nie zaśmiecać konsoli
    for k, debug_info in enumerate(debug_outputs[:10]): # Pokaż pierwsze 10
        print(f"Przykład {k+1}:")
        print(f"  Pytanie: {debug_info['input']}")
        print(f"  Prawidłowa odp (wartość): {debug_info['target_value']}")
        print(f"  Prawidłowa odp (string): {debug_info['target_str']}")
        print(f"  Przewidziana odp (wartość): {debug_info['predicted_val']:.4f}") # Wyświetl z większą precyzją
        print(f"  Przewidziana odp (string): {debug_info['predicted_str']}")
        print(f"  Poprawnie: {debug_info['is_correct']}")
        print("-" * 20)
    print("--------------------------------------------------")
    # KONIEC DODANYCH
    
    return predictions_decoded_values

def calculate_accuracy(predictions, targets):
    """
    Oblicza dokładność predykcji dla zadania arytmetycznego.
    """
    if not predictions:
        return 0
    
    correct = 0
    total_valid_predictions = 0 
    for pred_str, target_example in zip(predictions, targets):
        true_val_info = target_example[1] 
        true_val = float(true_val_info[0])
        
        try:
            pred_val = float(pred_str)
            
            # Tolerancja dla porównania zaokrąglonych wartości całkowitych
            if abs(round(pred_val) - round(true_val)) < 0.5: 
                correct += 1
            total_valid_predictions += 1 
        except ValueError:
            pass 

    return correct / total_valid_predictions if total_valid_predictions > 0 else 0.0

# --- GŁÓWNA PĘTLA TRENINGOWA ---
if __name__ == '__main__':
    device = torch.device("cpu") 

    print("Generating training examples...")
    train_examples = generate_improved_examples(n=5000, max_num=10000)
    val_examples = generate_improved_examples(n=500, max_num=10000)

    print("Building vocabulary...")
    vocab = build_vocab_from_examples(train_examples + val_examples)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    sample_val = 123.0
    sample_type = 'int'
    determined_numeric_feature_dim = len(number_embedding_features(sample_val, sample_type))
    print(f"Determined numeric feature dimension: {determined_numeric_feature_dim}")

    print("Initializing model...")
    model = ImprovedCrossEmbeddingModel(
        vocab_size=vocab_size,
        token_dim=128,          
        num_dim=128,            
        hidden=256,             
        num_layers=5,           # Zwiększono do 5 warstw
        dropout=0.1,
        feature_dim=determined_numeric_feature_dim 
    ).to(device)

    print(f"Model initialized with vocab size: {vocab_size}, numeric feature input dim: {determined_numeric_feature_dim}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20 * (5000 // 32)) 

    best_val_accuracy = -1.0
    num_epochs = 20
    batch_size = 32

    train_ce_losses = []
    train_align_losses = []
    train_num_losses = []
    val_accuracies = []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_stats = {'ce': 0.0, 'align': 0.0, 'num_loss': 0.0, 'total': 0.0, 'num_values': 0}
        
        random.shuffle(train_examples) 
        
        with tqdm(total=len(train_examples) // batch_size, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for i in range(0, len(train_examples), batch_size):
                batch = train_examples[i:i+batch_size]
                stats = train_step(model, optimizer, scheduler, batch, vocab, device)
                
                for key, val in stats.items():
                    if key in epoch_stats:
                        epoch_stats[key] += val
                
                pbar.set_postfix({
                    'ce': f"{stats['ce']:.3f}", 
                    'align': f"{stats['align']:.4f}", 
                    'num_loss': f"{stats['num_loss']:.2e}", 
                    'total': f"{stats['total']:.2e}",
                    'num_values': stats['num_values']
                })
                pbar.update(1)

        for key in ['ce', 'align', 'num_loss', 'total']:
            if epoch_stats['num_values'] > 0:
                epoch_stats[key] /= (epoch_stats['num_values'] / len(batch)) 
            else:
                epoch_stats[key] = 0.0

        print(f"\nEpoch {epoch+1} stats: CE: {epoch_stats['ce']:.4f}, Align: {epoch_stats['align']:.4f}, Num Loss: {epoch_stats['num_loss']:.4f}, Total: {epoch_stats['total']:.4f}")
        
        train_ce_losses.append(epoch_stats['ce'])
        train_align_losses.append(epoch_stats['align'])
        train_num_losses.append(epoch_stats['num_loss'])

        print("Evaluating on validation set...")
        val_predictions = evaluate(model, val_examples, vocab, device)
        val_accuracy = calculate_accuracy(val_predictions, val_examples)
        val_accuracies.append(val_accuracy)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

    print("\nTraining complete.")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_ce_losses, label='Train CE Loss')
    plt.plot(train_align_losses, label='Train Align Loss')
    plt.plot(train_num_losses, label='Train Num Loss')
    plt.title('Training Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
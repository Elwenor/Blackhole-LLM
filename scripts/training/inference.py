# File: inference.py
# Ten plik jest w większości poprawny, ale wprowadzam drobne poprawki
# dla spójności i niezawodności.

import torch
import re
import numpy as np

from blackhole.embedding import decode_number_from_features
from config import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, NUM_TOKEN,
    CAP_TOKEN, ALLCAPS_TOKEN, SPACE_TOKEN, SPECIAL_TOKENS
)

def predict_and_decode_answer(model, encoder_token_ids, encoder_numeric_features, encoder_attention_mask, vocab, device, max_decoding_len=128):
    """
    Generates an answer from the model given encoded input.
    Decodes predicted tokens and numerical features into a human-readable string.
    """
    model.eval()
    idx_to_token = {idx: token for token, idx in vocab.items()}
    num_token_id = vocab.get(NUM_TOKEN, vocab.get(UNK_TOKEN))
    bos_token_id = vocab.get(BOS_TOKEN, vocab.get(UNK_TOKEN))
    eos_token_id = vocab.get(EOS_TOKEN, vocab.get(UNK_TOKEN))
    
    # [POPRAWKA] Używamy `model.feature_dim`
    padded_feat_row = torch.full((model.feature_dim,), -2.0, dtype=torch.float32, device=device)

    batch_size = encoder_token_ids.size(0)
    if batch_size == 0:
        return [], []

    # Sprawdzenie, czy wejście nie jest całkowicie zamaskowane
    if torch.all(~encoder_attention_mask): # Nasza maska to True=uważaj, więc sprawdzamy all(False)
        placeholder = "<INPUT_FULLY_MASKED>"
        return [placeholder] * batch_size, [None] * batch_size

    # Inicjalizacja dekodera
    decoder_input_token_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    decoder_input_numeric_features = padded_feat_row.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Przechowujemy pełne sekwencje w trakcie generowania
    generated_token_sequences = [[] for _ in range(batch_size)]
    generated_feature_sequences = [[] for _ in range(batch_size)]
    
    # Flagi oznaczające, które sekwencje w batchu są już gotowe
    done_flags = [False] * batch_size

    with torch.no_grad():
        for _ in range(max_decoding_len):
            # Tworzymy maskę dla aktualnej długości dekodera
            decoder_attention_mask = torch.ones_like(decoder_input_token_ids, dtype=torch.bool, device=device)

            token_logits, num_feature_output = model(
                encoder_token_ids=encoder_token_ids,
                encoder_numeric_features=encoder_numeric_features,
                encoder_attention_mask=encoder_attention_mask,
                decoder_token_ids=decoder_input_token_ids,
                decoder_numeric_features_input=decoder_input_numeric_features,
                decoder_attention_mask=decoder_attention_mask
            )

            next_token_logits = token_logits[:, -1, :]
            next_num_features = num_feature_output[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1)

            # Dołączanie wygenerowanych tokenów i cech do wejścia na następny krok
            decoder_input_token_ids = torch.cat([decoder_input_token_ids, next_token_ids.unsqueeze(1)], dim=1)
            
            # [POPRAWKA] Bardziej zwięzła logika dodawania cech numerycznych
            next_step_features = torch.stack([
                next_num_features[b] if next_token_ids[b].item() == num_token_id else padded_feat_row
                for b in range(batch_size)
            ]).unsqueeze(1)
            decoder_input_numeric_features = torch.cat([decoder_input_numeric_features, next_step_features], dim=1)

            # Zbieranie wyników i sprawdzanie warunku stopu
            for i in range(batch_size):
                if not done_flags[i]:
                    token_id = next_token_ids[i].item()
                    if token_id == eos_token_id:
                        done_flags[i] = True
                    else:
                        generated_token_sequences[i].append(token_id)
                        # Zapisujemy cechy numeryczne dla każdego tokenu, nawet jeśli nie jest to <|num|>
                        # Ułatwi to późniejsze dekodowanie.
                        generated_feature_sequences[i].append(next_num_features[i].cpu().numpy())

            if all(done_flags):
                break

    # --- Dekodowanie wygenerowanych sekwencji ---
    final_decoded_answers = []
    final_predicted_numbers = []

    for i in range(batch_size):
        tokens = [idx_to_token.get(tok_id, UNK_TOKEN) for tok_id in generated_token_sequences[i]]
        features = generated_feature_sequences[i]
        
        decoded_numbers = [decode_number_from_features(feat) for feat in features]
        last_valid_number = next((num for num in reversed(decoded_numbers) if not np.isnan(num)), None)
        final_predicted_numbers.append(last_valid_number)
        
        # Rekonstrukcja tekstu
        predicted_answer_tokens = []
        num_idx = 0
        tok_idx = 0
        while tok_idx < len(tokens):
            token = tokens[tok_idx]
            if token == NUM_TOKEN:
                val = decoded_numbers[num_idx]
                if not np.isnan(val):
                    # Proste formatowanie
                    if abs(val - round(val)) < 1e-6:
                        predicted_answer_tokens.append(str(int(round(val))))
                    else:
                        predicted_answer_tokens.append(f"{val:.2f}")
                else:
                    predicted_answer_tokens.append("<NUM>") # Placeholder
                num_idx +=1
            elif token == CAP_TOKEN and tok_idx + 1 < len(tokens):
                predicted_answer_tokens.append(tokens[tok_idx+1].capitalize())
                tok_idx += 1
                num_idx += 1
            elif token == ALLCAPS_TOKEN and tok_idx + 1 < len(tokens):
                predicted_answer_tokens.append(tokens[tok_idx+1].upper())
                tok_idx += 1
                num_idx += 1
            elif token == SPACE_TOKEN:
                predicted_answer_tokens.append(' ')
            elif token not in [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                 predicted_answer_tokens.append(token)
            
            tok_idx += 1
            # [POPRAWKA] Upewnij się, że num_idx jest inkrementowany, chyba że to był specjalny token bez argumentu
            if token not in [CAP_TOKEN, ALLCAPS_TOKEN]:
                 num_idx +=1

        predicted_answer_raw = "".join(predicted_answer_tokens)
        predicted_answer_cleaned = re.sub(r'\s+([.,!?;:])', r'\1', predicted_answer_raw)
        predicted_answer_cleaned = re.sub(r'\s+', ' ', predicted_answer_cleaned).strip()
        final_decoded_answers.append(predicted_answer_cleaned)

    return final_decoded_answers, final_predicted_numbers
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers.activations import ACT2FN # Zmieniono import na stabilną ścieżkę
from transformers import PretrainedConfig

# Klasa konfiguracyjna modelu, dziedzicząca z PretrainedConfig
class BlackholeConfig(PretrainedConfig):
    model_type = "blackhole" # Unikalny typ modelu

    def __init__(
        self,
        vocab_size: int = 50265,  # Domyślny rozmiar słownika, np. z RoBERTa/GPT-2
        hidden_size: int = 768,   # Wymiar osadzeń (d_model w transformerze)
        num_hidden_layers: int = 12, # Domyślna liczba warstw Bert
        num_attention_heads: int = 12, # Domyślna liczba głów uwagi Bert
        intermediate_size: int = 3072, # Domyślny rozmiar warstwy pośredniej Bert
        max_position_embeddings: int = 512, # Maksymalna długość sekwencji
        type_vocab_size: int = 2, # Typy tokenów (dla sekwencji A/B)
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        hidden_act: str = "gelu",
        pad_token_id: int = 1,    # Ustaw na rzeczywiste ID paddingu z Twojego tokenizer'a
        num_token_id: int = 5,    # Ustaw na rzeczywiste ID tokenu [NUM] z Twojego tokenizer'a
        classifier_dropout: Optional[float] = None, # Parametr dla klasyfikacji sekwencji
        # Nowe parametry dla zaawansowanych cech numerycznych
        numeric_feature_dims: dict = { # Wymiary dla poszczególnych typów cech (suma musi wynieść 96)
            # HEAVY LAYERS / Highly Informative Features (64 + 20 = 84 cechy)
            "float64_binary_repr": 64,      # 64-bitowa reprezentacja IEEE 754 (najcięższa)
            "digit_pos_0": 10,              # One-hot dla cyfry jedności (0-9)
            "digit_pos_1": 10,              # One-hot dla cyfry dziesiątek (0-9)

            # LIGHT LAYERS / Simpler Informative Features (5 + 7 = 12 cech)
            "log_value": 1,                 # Logarytm z wartości bezwzględnej
            "sign": 1,                      # Znak liczby (-1, 0, 1)
            "exponent_base10": 1,           # Wykładnik potęgi 10 (rząd wielkości)
            "num_total_digits": 1,          # Całkowita liczba cyfr (przed i po przecinku)
            "num_decimal_places": 1,        # Liczba miejsc po przecinku

            "is_integer_flag": 1,           # Czy liczba jest całkowita
            "is_positive_flag": 1,          # Czy liczba jest dodatnia
            "is_zero_flag": 1,              # Czy liczba jest równa 0
            "is_negative_flag": 1,          # Czy liczba jest ujemna
            "is_power_of_2_flag": 1,        # Czy jest potęgą 2
            "format_type_int": 1,           # Czy format to integer (one-hot)
            "format_type_float": 1,         # Czy format to float (one-hot)
            # Suma wszystkich cech: 64 + 10 + 10 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 96
        },
        numeric_projection_intermediate_size_ratio: float = 0.5, # np. 0.5 * hidden_size
        numeric_embedding_fusion_type: str = "gating", # "add", "concat", "gating"
        numeric_heavy_feature_freeze: bool = False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers # Ustawienie parametru
        self.num_attention_heads = num_attention_heads # Ustawienie parametru
        self.intermediate_size = intermediate_size # Ustawienie parametru
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.num_token_id = num_token_id
        self.classifier_dropout = classifier_dropout

        # Oblicz całkowitą liczbę wejściowych cech numerycznych
        self.numeric_feature_dims = numeric_feature_dims
        self.numeric_input_features = sum(numeric_feature_dims.values())
        if self.numeric_input_features != 96:
            # Zmieniono na warnings.warn, aby nie przerywać działania, ale ostrzegać
            import warnings
            warnings.warn(
                f"Suma cech numerycznych powinna wynosić 96. Obecnie wynosi: {self.numeric_input_features}. "
                "Sprawdź definicję 'numeric_feature_dims' w BlackholeConfig. Kontynuowanie mimo to."
            )

        self.numeric_projection_intermediate_size = int(hidden_size * numeric_projection_intermediate_size_ratio)
        self.numeric_embedding_fusion_type = numeric_embedding_fusion_type
        self.numeric_heavy_feature_freeze = numeric_heavy_feature_freeze

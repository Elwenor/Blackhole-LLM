import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers.models.bert.modeling_bert import ACT2FN
from transformers import PretrainedConfig

# Klasa konfiguracyjna modelu, dziedzicząca z PretrainedConfig
class BlackholeConfig(PretrainedConfig):
    model_type = "blackhole" # Unikalny typ modelu

    def __init__(
        self,
        vocab_size: int = 50265,  # Domyślny rozmiar słownika, np. z RoBERTa/GPT-2
        hidden_size: int = 768,   # Wymiar osadzeń (d_model w transformerze)
        max_position_embeddings: int = 512, # Maksymalna długość sekwencji
        type_vocab_size: int = 2, # Typy tokenów (dla sekwencji A/B)
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        hidden_act: str = "gelu", # <-- DODAJ TĘ LINIĘ
        pad_token_id: int = 1,    # Ustaw na rzeczywiste ID paddingu z Twojego tokenizer'a
        num_token_id: int = 5,    # Ustaw na rzeczywiste ID tokenu [NUM] z Twojego tokenizer'a
        # Nowe parametry dla zaawansowanych cech numerycznych
        numeric_feature_dims: dict = { # Wymiary dla poszczególnych typów cech (suma musi wynieść 96)
            # HEAVY LAYERS / Highly Informative Features (64 + 20 = 84 cechy)
            "float64_binary_repr": 64,     # 64-bitowa reprezentacja IEEE 754 (najcięższa)
            "digit_pos_0": 10,             # One-hot dla cyfry jedności (0-9)
            "digit_pos_1": 10,             # One-hot dla cyfry dziesiątek (0-9)

            # LIGHT LAYERS / Simpler Informative Features (5 + 7 = 12 cech)
            "log_value": 1,                # Logarytm z wartości bezwzględnej
            "sign": 1,                     # Znak liczby (-1, 0, 1)
            "exponent_base10": 1,          # Wykładnik potęgi 10 (rząd wielkości)
            "num_total_digits": 1,         # Całkowita liczba cyfr (przed i po przecinku)
            "num_decimal_places": 1,       # Liczba miejsc po przecinku

            "is_integer_flag": 1,          # Czy liczba jest całkowita
            "is_positive_flag": 1,         # Czy liczba jest dodatnia
            "is_zero_flag": 1,             # Czy liczba jest równa 0
            "is_negative_flag": 1,         # Czy liczba jest ujemna
            "is_power_of_2_flag": 1,       # Czy jest potęgą 2
            "format_type_int": 1,          # Czy format to integer (one-hot)
            "format_type_float": 1,        # Czy format to float (one-hot)
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
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act # <-- DODAJ TĘ LINIĘ
        self.num_token_id = num_token_id

        # Oblicz całkowitą liczbę wejściowych cech numerycznych
        self.numeric_feature_dims = numeric_feature_dims
        self.numeric_input_features = sum(numeric_feature_dims.values())
        if self.numeric_input_features != 96:
            raise ValueError(
                f"Suma cech numerycznych musi wynosić 96. Obecnie wynosi: {self.numeric_input_features}. "
                "Sprawdź definicję 'numeric_feature_dims' w BlackholeConfig."
            )

        self.numeric_projection_intermediate_size = int(hidden_size * numeric_projection_intermediate_size_ratio)
        self.numeric_embedding_fusion_type = numeric_embedding_fusion_type
        self.numeric_heavy_feature_freeze = numeric_heavy_feature_freeze


class BlackholeEmbeddings(nn.Module):
    def __init__(self, config: BlackholeConfig):
        super().__init__()
        self.config = config # Zapisujemy config dla dostępu do parametrów

        # Standardowe osadzenia tokenów (słów)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # Osadzenia pozycji
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # Osadzenia typu tokenu (dla segmentów, np. A/B w BERT)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # Warstwy normalizacji i dropoutu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # --- Zaawansowana warstwa do osadzania liczb ---
        # Ta sieć MLP będzie przekształcać rozbudowane cechy numeryczne w wektor osadzeń.
        self.numeric_embedding_projection = nn.Sequential(
            nn.Linear(config.numeric_input_features, config.numeric_projection_intermediate_size),
            ACT2FN["gelu"], # Aktywacja GELU z Hugging Face
            nn.Linear(config.numeric_projection_intermediate_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # Normalizacja po projekcji
        )

        # Mechanizm bramkowania (gating) dla fuzji osadzeń numerycznych z tekstowymi
        self.numeric_embedding_fusion_type = config.numeric_embedding_fusion_type
        if self.numeric_embedding_fusion_type == "gating":
            self.numeric_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.gate_activation = nn.Sigmoid()
        elif self.numeric_embedding_fusion_type == "add":
            pass # Fuzja przez dodawanie, nie potrzebuje dodatkowych warstw
        elif self.numeric_embedding_fusion_type == "concat":
            self.concat_projection = nn.Linear(config.hidden_size * 2, config.hidden_size)
        else:
            raise ValueError(f"Nieznany typ fuzji numerycznej: {config.numeric_embedding_fusion_type}. Oczekiwano 'add', 'concat' lub 'gating'.")

        self.num_token_id = config.num_token_id
        # Wartość paddingu dla numeric_values (konieczne, by tokenizer to zwracał!)
        self.numeric_pad_value = float('nan')

        # Bufor do przechowywania position_ids, dla efektywności (zgodnie z HF)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # --- Zamrażanie wagi dla 'ciężkich' cech numerycznych, jeśli włączone ---
        if config.numeric_heavy_feature_freeze:
            for param in self.numeric_embedding_projection.parameters():
                param.requires_grad = False
            print("Wagi numerycznej warstwy projekcyjnej zostały zamrożone.")


    def _get_numeric_features(self, values: torch.Tensor, formats: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Przetwarza surowe wartości liczbowe i ich formaty na wejściowe cechy numeryczne
        dla `numeric_embedding_projection`.
        Wewnętrznie używa float64 dla maksymalnej precyzji podczas ekstrakcji cech,
        ale zwraca float32, aby było kompatybilne z pozostałą częścią modelu.

        Args:
            values (torch.Tensor): Tensor wartości numerycznych (tylko faktyczne liczby).
                                    Powinien być typu torch.float64 dla precyzji binarnej.
            formats (torch.Tensor, optional): Tensor z ID formatów numerycznych
                                            (0: int, 1: float, 2: scientific, 3: hexadecimal).
        Returns:
            torch.Tensor: Złączone cechy numeryczne, TYPU torch.float32.
                            Kształt: (num_numeric_tokens, total_features).
        """
        features_list = []
        device = values.device

        # WAŻNE: Upewniamy się, że wartości są w float64 dla dokładnej reprezentacji bitowej
        if values.dtype != torch.float64:
            values = values.double()

        # --- HEAVY LAYERS / Highly Informative Features (84 cechy) ---

        # 1. Pełna 64-bitowa reprezentacja IEEE 754 (float64_binary_repr) - 64 cechy
        if self.config.numeric_feature_dims.get("float64_binary_repr", 0) > 0:
            num_bits = self.config.numeric_feature_dims["float64_binary_repr"]

            # Reinterpretacja bitów float jako int64
            long_values = values.view(torch.int64)

            # Tworzymy maski bitowe: [1, 2, 4, 8, ..., 2^63]
            bit_indices = torch.arange(num_bits, device=device, dtype=torch.int64)
            masks = (1 << bit_indices)

            # Wykonujemy bitowe AND i sprawdzamy, czy bit jest ustawiony
            binary_features = ((long_values.unsqueeze(-1) & masks) > 0).float() # Konwertujemy na float32
            features_list.append(binary_features)

        # 2. Cechy Pozycji Cyfr (20 cech)
        abs_values = torch.abs(values)

        # 2.1. Cyfra jedności (digit_pos_0) - 10 cech (one-hot)
        if self.config.numeric_feature_dims.get("digit_pos_0", 0) > 0:
            units_digit = (abs_values.floor() % 10).long()
            units_digit = torch.clamp(units_digit, 0, 9)
            one_hot_units = torch.nn.functional.one_hot(units_digit, num_classes=10).float()
            features_list.append(one_hot_units)

        # 2.2. Cyfra dziesiątek (digit_pos_1) - 10 cech (one-hot)
        if self.config.numeric_feature_dims.get("digit_pos_1", 0) > 0:
            tens_digit = ((abs_values.floor() // 10) % 10).long()
            tens_digit = torch.clamp(tens_digit, 0, 9)
            one_hot_tens = torch.nn.functional.one_hot(tens_digit, num_classes=10).float()
            features_list.append(one_hot_tens)

        # --- LIGHT LAYERS / Simpler Informative Features (12 cech) ---

        # 3. Cechy Skalarne (5 cech)

        # 3.1. Logarytm z wartości bezwzględnej (log_value)
        if self.config.numeric_feature_dims.get("log_value", 0) > 0:
            log_abs_values = torch.log(torch.abs(values) + 1e-6)
            features_list.append(log_abs_values.unsqueeze(-1).float())

        # 3.2. Znak liczby (sign)
        if self.config.numeric_feature_dims.get("sign", 0) > 0:
            signs = torch.sign(values)
            signs[values == 0.0] = 0.0
            features_list.append(signs.unsqueeze(-1).float())

        # 3.3. Wykładnik potęgi 10 (exponent_base10)
        if self.config.numeric_feature_dims.get("exponent_base10", 0) > 0:
            exponents = torch.where(
                torch.abs(values) > 1e-6,
                torch.floor(torch.log10(torch.abs(values))),
                torch.tensor(0.0, device=device, dtype=values.dtype)
            )
            features_list.append(exponents.unsqueeze(-1).float())

        # 3.4. Całkowita liczba cyfr w liczbie (num_total_digits)
        if self.config.numeric_feature_dims.get("num_total_digits", 0) > 0:
            total_digits_tensor = torch.tensor([
                sum(1 for char in str(val.item()).replace('.', '').replace('-', '').lower().split('e')[0] if char.isdigit())
                for val in values
            ], dtype=torch.float32, device=device)
            features_list.append(total_digits_tensor.unsqueeze(-1))


        # 3.5. Liczba miejsc po przecinku (num_decimal_places)
        if self.config.numeric_feature_dims.get("num_decimal_places", 0) > 0:
            decimal_places_tensor = torch.tensor([
                (lambda s: len(s.split('.')[-1]) if '.' in s else 0)(str(val.item()).lower().split('e')[0])
                for val in values
            ], dtype=torch.float32, device=device)
            features_list.append(decimal_places_tensor.unsqueeze(-1))

        # 4. Flagi Logiczne/Semantyczne (7 cech)

        # 4.1. Czy liczba jest całkowita (is_integer_flag)
        if self.config.numeric_feature_dims.get("is_integer_flag", 0) > 0:
            is_integer = (torch.abs(values) == torch.abs(values).floor()).float()
            features_list.append(is_integer.unsqueeze(-1))

        # 4.2. Czy liczba jest dodatnia (is_positive_flag)
        if self.config.numeric_feature_dims.get("is_positive_flag", 0) > 0:
            is_positive = (values > 0).float()
            features_list.append(is_positive.unsqueeze(-1))

        # 4.3. Czy liczba jest równa 0 (is_zero_flag)
        if self.config.numeric_feature_dims.get("is_zero_flag", 0) > 0:
            is_zero = (values == 0.0).float()
            features_list.append(is_zero.unsqueeze(-1))

        # 4.4. Czy liczba jest ujemna (is_negative_flag)
        if self.config.numeric_feature_dims.get("is_negative_flag", 0) > 0:
            is_negative = (values < 0).float()
            features_list.append(is_negative.unsqueeze(-1))

        # 4.5. Czy liczba jest potęgą 2 (is_power_of_2_flag)
        if self.config.numeric_feature_dims.get("is_power_of_2_flag", 0) > 0:
            is_positive_integer = (values == values.floor()) & (values > 0)

            log2_val_safe = torch.full_like(values, float('nan'), dtype=torch.float64)
            valid_indices = is_positive_integer.nonzero(as_tuple=True)[0]

            if valid_indices.numel() > 0:
                log2_val_safe[valid_indices] = torch.log2(values[valid_indices])

            is_log2_integer = (log2_val_safe == log2_val_safe.floor())

            is_power_of_2 = (is_positive_integer & is_log2_integer).float()
            features_list.append(is_power_of_2.unsqueeze(-1))

        # 4.6 i 4.7. Typ formatu (format_type_int, format_type_float)
        if formats is not None:
            if self.config.numeric_feature_dims.get("format_type_int", 0) > 0:
                is_format_int = (formats == 0).float()
                features_list.append(is_format_int.unsqueeze(-1))
            if self.config.numeric_feature_dims.get("format_type_float", 0) > 0:
                is_format_float = (formats == 1).float()
                features_list.append(is_format_float.unsqueeze(-1))


        # Łączymy wszystkie cechy w jeden tensor
        # WAŻNE: Konwertujemy wszystkie cechy na float32 przed złączeniem i zwróceniem,
        # aby były zgodne z oczekiwanym dtype warstwy projekcji.
        combined_features = torch.cat([f.to(torch.float32) for f in features_list], dim=-1)

        if combined_features.shape[-1] != self.config.numeric_input_features:
            raise ValueError(
                f"Niezgodność liczby wygenerowanych cech ({combined_features.shape[-1]}) "
                f"z oczekiwaną w konfiguracji ({self.config.numeric_input_features}). "
                f"Upewnij się, że wszystkie klucze w `numeric_feature_dims` są obsługiwane w `_get_numeric_features` "
                f"i że ich sumy się zgadzają z {self.config.numeric_input_features}."
            )

        return combined_features


    def forward(
        self,
        input_ids: torch.Tensor,
        numeric_values: torch.Tensor, # Zawiera torch.nan dla pozycji nienumerycznych
        numeric_formats: Optional[torch.Tensor] = None, # NOWE: Tensor z typami formatów (0:int, 1:float, 2:scientific, -1:padding)
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor: # Zwracamy JEDEN tensor, który łączy tekstowe i numeryczne

        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device

        # --- Standardowe osadzenia tekstowe ---
        if inputs_embeds is None:
            text_word_embeddings = self.word_embeddings(input_ids)
        else:
            text_word_embeddings = inputs_embeds

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length].expand_as(input_ids)
        text_position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        text_token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Sumowanie osadzeń tekstowych (bazowe osadzenie), upewnij się, że są float32
        text_embeddings = text_word_embeddings + text_position_embeddings + text_token_type_embeddings
        text_embeddings = text_embeddings.to(torch.float32)

        # --- Generowanie osadzeń numerycznych dla tokenów [NUM] ---
        is_num_token_mask = (input_ids == self.num_token_id)
        has_numeric_value_mask = ~torch.isnan(numeric_values)

        active_numeric_positions_mask = is_num_token_mask & has_numeric_value_mask

        # Inicjalizacja tensora dla osadzeń numerycznych (zerami), typu float32
        numeric_embeds_for_fusion = torch.zeros(
            input_shape[0], input_shape[1], self.config.hidden_size,
            device=device, dtype=torch.float32 # Tutaj powinno być float32
        )

        if active_numeric_positions_mask.any():
            # Pobierz wartości i formaty tylko dla aktywowanych pozycji numerycznych
            # numeric_values mogą być torch.float64, ale _get_numeric_features to obsłuży
            actual_numeric_values = numeric_values[active_numeric_positions_mask]
            actual_numeric_formats = None
            if numeric_formats is not None:
                actual_numeric_formats = numeric_formats[active_numeric_positions_mask]

            # Generuj zaawansowane cechy numeryczne (zwróci float32)
            processed_numeric_features = self._get_numeric_features(actual_numeric_values, actual_numeric_formats)

            # Przekształć cechy w osadzenia numeryczne za pomocą MLP (oczekuje float32)
            projected_numeric_embeds = self.numeric_embedding_projection(processed_numeric_features)

            # Umieść wygenerowane osadzenia z powrotem w tensorze `numeric_embeds_for_fusion`
            numeric_embeds_for_fusion[active_numeric_positions_mask] = projected_numeric_embeds

        # --- Fuzja osadzeń tekstowych i numerycznych ---
        # Oba tensory (text_embeddings i numeric_embeds_for_fusion) są teraz float32.

        if self.numeric_embedding_fusion_type == "gating":
            combined_for_gate = torch.cat((text_embeddings, numeric_embeds_for_fusion), dim=-1)
            # numeric_gate i gate_activation są teraz w float32
            gate = self.gate_activation(self.numeric_gate(combined_for_gate))
            final_embeddings = (1 - gate) * text_embeddings + gate * numeric_embeds_for_fusion

        elif self.numeric_embedding_fusion_type == "add":
            final_embeddings = text_embeddings + numeric_embeds_for_fusion

        elif self.numeric_embedding_fusion_type == "concat":
            concatenated_embeddings = torch.cat((text_embeddings, numeric_embeds_for_fusion), dim=-1)
            # concat_projection jest teraz w float32
            final_embeddings = self.concat_projection(concatenated_embeddings)

        else:
            final_embeddings = text_embeddings

        # --- Finalna normalizacja i dropout dla całego tensora osadzeń ---
        final_embeddings = self.LayerNorm(final_embeddings)
        final_embeddings = self.dropout(final_embeddings)

        return final_embeddings
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
# from transformers.activations import ACT2FN
# Importujemy ACT2FN z modułu transformers.models.bert.modeling_bert
# aby uniknąć zależności od całego transformers.activations
from transformers.models.bert.modeling_bert import ACT2FN
from transformers import PretrainedConfig

# Aby upewnić się, że BlackholeTokenizer jest dostępne zgodnie z Twoją ścieżką
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))
from blackhole.tokenizer_hugging_face import BlackholeTokenizer # Zachowuję Twój import

# Klasa konfiguracyjna modelu, dziedzicząca z PretrainedConfig
# Powinna zawierać wszystkie niezbędne parametry dla BlackholeEmbeddings
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
        pad_token_id: int = 1,    # Ustaw na rzeczywiste ID paddingu z Twojego tokenizer'a
        num_token_id: int = 5,    # Ustaw na rzeczywiste ID tokenu [NUM] z Twojego tokenizer'a
        # Nowe parametry dla zaawansowanych cech numerycznych
        numeric_feature_dims: dict = { # Wymiary dla poszczególnych typów cech
            "log_value": 1,             # Logarytm z wartości bezwzględnej
            "sign": 1,                  # Znak liczby
            "exponent": 1,              # Wykładnik potęgi 10 (rzędu wielkości)
            "binary_representation": 16, # Uproszczona binarna reprezentacja (np. 16 bitów)
            "format_type": 3,           # One-hot encoding dla 3 typów formatów (int, float, scientific)
        },
        numeric_projection_intermediate_size_ratio: float = 0.5, # np. 0.5 * hidden_size
        numeric_embedding_fusion_type: str = "gating", # "add", "concat", "gating"
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_token_id = num_token_id

        # Oblicz całkowitą liczbę wejściowych cech numerycznych
        self.numeric_feature_dims = numeric_feature_dims
        self.numeric_input_features = sum(numeric_feature_dims.values())

        self.numeric_projection_intermediate_size = int(hidden_size * numeric_projection_intermediate_size_ratio)
        self.numeric_embedding_fusion_type = numeric_embedding_fusion_type


class BlackholeEmbeddings(nn.Module):
    """
    Warstwa osadzeń dla modelu Blackhole-LLM, obsługująca dwa strumienie:
    - Strumień tekstowy (tokeny BPE + specjalne tokeny)
    - Strumień numeryczny (rzeczywiste wartości liczbowe + ich cechy)

    Łączy osadzenia tokenów, osadzenia pozycji oraz specjalne osadzenia numeryczne,
    przygotowując je do dalszego przetwarzania w architekturze Transformera.
    """
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
            # Bramka bierze konkatenację osadzeń tekstowych i numerycznych w pozycji [NUM]
            self.numeric_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.gate_activation = nn.Sigmoid()
        elif self.numeric_embedding_fusion_type == "add":
            pass # Fuzja przez dodawanie, nie potrzebuje dodatkowych warstw
        elif self.numeric_embedding_fusion_type == "concat":
            # Jeśli łączymy, hidden_size musi być dostosowane w dalszych warstwach, lub musimy mieć projekcję
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

    def _get_numeric_features(self, values: torch.Tensor, formats: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Przetwarza surowe wartości liczbowe i ich formaty na wejściowe cechy numeryczne
        dla `numeric_embedding_projection`.

        Args:
            values (torch.Tensor): Tensor wartości numerycznych (tylko faktyczne liczby).
            formats (torch.Tensor, optional): Tensor z ID formatów numerycznych
                                              (0: int, 1: float, 2: scientific, -1: padding/nan).
        Returns:
            torch.Tensor: Złączone cechy numeryczne. Kształt: (num_numeric_tokens, total_features).
        """
        features_list = []
        device = values.device

        # 1. Logarytm z wartości bezwzględnej (log_value)
        if self.config.numeric_feature_dims.get("log_value", 0) > 0:
            log_abs_values = torch.log(torch.abs(values) + 1e-6) # Dodajemy mały epsilon dla stabilności
            features_list.append(log_abs_values.unsqueeze(-1))

        # 2. Znak liczby (sign)
        if self.config.numeric_feature_dims.get("sign", 0) > 0:
            signs = torch.sign(values)
            signs[values == 0.0] = 0.0 # Upewniamy się, że dla 0 znak to 0
            features_list.append(signs.unsqueeze(-1))

        # 3. Wykładnik potęgi 10 (exponent)
        if self.config.numeric_feature_dims.get("exponent", 0) > 0:
            # Dla floatów (lub ogólnie liczb), możemy użyć log10 do wyznaczenia rzędu wielkości
            # obsługa log10(0) i wartości ujemnych
            exponents = torch.where(
                torch.abs(values) > 1e-6, # Tylko dla niezerowych wartości
                torch.floor(torch.log10(torch.abs(values))),
                torch.tensor(0.0, device=device) # Ustaw 0 dla bardzo małych/zerowych wartości
            )
            features_list.append(exponents.unsqueeze(-1))

        # 4. Uproszczona Binarna Reprezentacja (binary_representation)
        # To jest uproszczona implementacja. Prawdziwa binarna dla floatów jest bardziej złożona.
        # Może to być np. sinusoidalne osadzenie liczby, lub "hash" bitowy.
        if self.config.numeric_feature_dims.get("binary_representation", 0) > 0:
            # Możesz użyć np. sinusoidalnych osadzeń dla wartości numerycznej jako "proxy" dla binarnej
            # lub prostej projekcji na N bitów
            num_bits = self.config.numeric_feature_dims["binary_representation"]
            
            # Poniżej prosta próba rzutowania na bity. Dla floatów to jest bardzo nieidealne.
            # Lepszym podejściem byłoby np. `torch.fmod(values * 2**i, 1.0)` do "rozkładania" ułamków.
            # Albo użycie funkcji kwantyzacji do mapowania liczb na zbiór "bucketów" i ich osadzania.
            # Na potrzeby tego kodu, będziemy generować "bity" z hash'a lub rzutowania.
            
            # Jeśli masz liczby całkowite, możesz je rzutować na int i potem na bity:
            # integer_part = values.long()
            # bit_features = ((integer_part.unsqueeze(-1) >> torch.arange(num_bits, device=device)) & 1).float()
            # features_list.append(bit_features)
            
            # Dla ogólnych floatów, to jest miejsce na innowację.
            # Na potrzeby tego testu, stworzymy pewne 'pseudo-bity' z wartości.
            # Np. mapowanie wartości na przedział [0, 1] i potem na bity
            # (values - min_val) / (max_val - min_val)
            # Na razie jako placeholder: generujemy losowe bity (w rzeczywistości musiałoby być zdeterminowane przez wartość)
            pseudo_binary_features = torch.randn(values.shape[0], num_bits, device=device) # Zastąp to rzeczywistą logiką
            features_list.append(pseudo_binary_features)


        # 5. Typ formatu (format_type)
        if formats is not None and self.config.numeric_feature_dims.get("format_type", 0) > 0:
            # Formaty: 0=int, 1=float, 2=scientific (lub inne, jeśli zdefiniujesz więcej)
            # Upewnij się, że formats jest odpowiednio przeskalowany/zakodowany z tokenizera.
            # Formaty -1 (padding) powinny być ignorowane w `_get_numeric_features`.
            # Ta funkcja _get_numeric_features jest wywoływana TYLKO dla `actual_numeric_values`
            # więc `formats` tutaj nie powinno zawierać -1.
            one_hot_formats = torch.nn.functional.one_hot(
                formats.long(), num_classes=self.config.numeric_feature_dims["format_type"]
            ).float()
            features_list.append(one_hot_formats)

        # Łączymy wszystkie cechy w jeden tensor
        combined_features = torch.cat(features_list, dim=-1)
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
        """
        Przetwarza wejścia i generuje finalne osadzenia wejściowe dla Transformera.

        Args:
            input_ids (torch.Tensor): Tensory z ID tokenów ze strumienia tekstowego.
                                      Kształt: (batch_size, sequence_length).
            numeric_values (torch.Tensor): Tensory z rzeczywistymi wartościami liczbowymi.
                                            Kształt: (batch_size, sequence_length).
                                            Zawiera self.numeric_pad_value (torch.nan) dla pozycji nienumerycznych.
            numeric_formats (torch.Tensor, optional): Tensory z ID formatów numerycznych.
                                                     Kształt: (batch_size, sequence_length).
                                                     Zawiera -1 dla pozycji nienumerycznych.
            token_type_ids (torch.Tensor, optional): Tensory z ID typów tokenów (dla segmentów A/B).
            position_ids (torch.Tensor, optional): Tensory z ID pozycji.
            inputs_embeds (torch.Tensor, optional): Bezpośrednio podane osadzenia wejściowe.
            past_key_values_length (int): Długość poprzednich kluczy/wartości dla generacji.

        Returns:
            torch.Tensor: Finalne osadzenia wejściowe dla Transformera.
                          Kształt: (batch_size, sequence_length, hidden_size).
        """
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

        # Sumowanie osadzeń tekstowych (bazowe osadzenie)
        text_embeddings = text_word_embeddings + text_position_embeddings + text_token_type_embeddings

        # --- Generowanie osadzeń numerycznych dla tokenów [NUM] ---
        # Maska dla pozycji, które są tokenami [NUM] I mają rzeczywiste wartości liczbowe (nie NaN)
        # `is_num_token_mask` identyfikuje, gdzie w input_ids jest [NUM]
        is_num_token_mask = (input_ids == self.num_token_id)
        # `has_numeric_value_mask` sprawdza, czy w `numeric_values` nie ma NaN w tych pozycjach
        has_numeric_value_mask = ~torch.isnan(numeric_values)
        
        # Ostateczna maska dla pozycji, gdzie faktycznie powinniśmy generować numeryczne embeddingi
        # (tylko tam, gdzie jest token [NUM] i przypisana do niego sensowna wartość numeryczna)
        active_numeric_positions_mask = is_num_token_mask & has_numeric_value_mask

        # Inicjalizacja tensora dla osadzeń numerycznych (zerami).
        # Będą one niezerowe tylko w miejscach wskazanych przez `active_numeric_positions_mask`.
        numeric_embeds_for_fusion = torch.zeros(
            input_shape[0], input_shape[1], self.config.hidden_size,
            device=device, dtype=text_embeddings.dtype
        )

        if active_numeric_positions_mask.any():
            # Pobierz wartości i formaty tylko dla aktywowanych pozycji numerycznych
            actual_numeric_values = numeric_values[active_numeric_positions_mask]
            actual_numeric_formats = None
            if numeric_formats is not None:
                actual_numeric_formats = numeric_formats[active_numeric_positions_mask]

            # Generuj zaawansowane cechy numeryczne
            processed_numeric_features = self._get_numeric_features(actual_numeric_values, actual_numeric_formats)

            # Przekształć cechy w osadzenia numeryczne
            projected_numeric_embeds = self.numeric_embedding_projection(processed_numeric_features)

            # Umieść wygenerowane osadzenia z powrotem w tensorze `numeric_embeds_for_fusion`
            numeric_embeds_for_fusion[active_numeric_positions_mask] = projected_numeric_embeds

        # --- Fuzja osadzeń tekstowych i numerycznych ---
        # Tutaj następuje inteligentne łączenie informacji tekstowych i numerycznych.
        if self.numeric_embedding_fusion_type == "gating":
            # Mechanizm bramkowania: model dynamicznie decyduje, jak dużo informacji numerycznej włączyć.
            # Konkatenuje oba typy embeddingów dla tokenów [NUM] i uczy się wagi.
            
            # Przygotowujemy input dla bramki, łącząc text_embeddings i numeric_embeds_for_fusion
            # Tylko dla tych pozycji, które są aktywne numerycznie
            
            # Tworzymy tensor, który będzie zawierał połączone embeddingi tylko dla aktywnych pozycji numerycznych
            # Pozostałe miejsca będą miały zera lub inną bazową wartość, aby bramka nie była uczona na nich.
            
            # W bardziej zaawansowanych implementacjach, bramka może być uczona dla wszystkich tokenów,
            # lub tylko dla tych, które są [NUM]. Tutaj uczymy bramkę tylko dla [NUM] tokenów.
            
            # Jeśli bramka działa na CAŁYM tensorze (batch_size, seq_len, hidden_size * 2)
            # a numeric_embeds_for_fusion ma zera tam, gdzie nie ma liczb, to jest to ok.
            
            # Wartość bramki dla pozycji, które nie są tokenami [NUM], będzie bliska 0 lub 1
            # w zależności od tego, jak model nauczy się je traktować.
            
            combined_for_gate = torch.cat((text_embeddings, numeric_embeds_for_fusion), dim=-1)
            gate = self.gate_activation(self.numeric_gate(combined_for_gate))
            
            # Finalne osadzenie to ważona suma osadzeń tekstowych i numerycznych
            # Dla pozycji, które nie są [NUM], numeric_embeds_for_fusion jest zerem,
            # więc gate * 0 jest 0. Wtedy (1 - gate) * text_embeddings jest dominujące.
            final_embeddings = (1 - gate) * text_embeddings + gate * numeric_embeds_for_fusion

        elif self.numeric_embedding_fusion_type == "add":
            # Proste dodawanie: informacja numeryczna dodawana do tekstowej.
            # Numeric_embeds_for_fusion ma zera tam, gdzie nie ma [NUM] tokenu.
            final_embeddings = text_embeddings + numeric_embeds_for_fusion

        elif self.numeric_embedding_fusion_type == "concat":
            # Konkatenacja, a następnie projekcja do docelowego hidden_size.
            # Wymaga, aby self.concat_projection istniało.
            concatenated_embeddings = torch.cat((text_embeddings, numeric_embeds_for_fusion), dim=-1)
            final_embeddings = self.concat_projection(concatenated_embeddings)
        
        else: # Powinno być już obsłużone w __init__, ale dla bezpieczeństwa
            final_embeddings = text_embeddings # Domyślny fallback

        # --- Finalna normalizacja i dropout dla całego tensora osadzeń ---
        final_embeddings = self.LayerNorm(final_embeddings)
        final_embeddings = self.dropout(final_embeddings)

        return final_embeddings
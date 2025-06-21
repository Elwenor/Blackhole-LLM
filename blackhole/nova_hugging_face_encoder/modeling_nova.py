import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers.activations import ACT2FN # Zmieniono import na stabilną ścieżkę
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bert.modeling_bert import ( # Używamy komponentów BERT-a jako bazę
    BertSelfAttention,
    BertSelfOutput,
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertLayer,
    BertEncoder,
    BertPooler,
)

from .configuration_nova import BlackholeConfig

# Importujemy BlackholeEmbeddings z Twojego istniejącego pliku hugging_embedding.py
from blackhole.embadding_hugging_face.hugging_embedding import BlackholeEmbeddings as OriginalBlackholeEmbeddings

logger = logging.get_logger(__name__)


# DOCSTRING
BLACKHOLE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BlackholeTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        numeric_values (`torch.DoubleTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Tensor containing the numeric values associated with tokens.
            This tensor should contain `float('nan')` for non-numeric tokens.
            This is processed by the BlackholeEmbeddings layer.
        numeric_formats (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Tensor containing integer IDs representing the format of numeric values (e.g., 0 for integer, 1 for float).
            This tensor should contain `-1` for non-numeric tokens.
            This is processed by the BlackholeEmbeddings layer.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Opcjonalnie, zamiast przekazywać `input_ids`, możesz bezpośrednio przekazać osadzone wejście. Jest to przydatne,
            jeśli chcesz mieć większą kontrolę nad tym, jak generowane są Twoje osadzenia wejściowe.

            Jeśli `inputs_embeds` jest przekazane, to `input_ids` i `token_type_ids` (jeśli określono) są ignorowane.
        output_attentions (`bool`, *optional*):
            Czy zwracać tensory uwagi wszystkich warstw uwagi. Zobacz `attentions` w zwróconych
            tensorach, aby uzyskać więcej szczegółów.
        output_hidden_states (`bool`, *optional*):
            Czy zwracać ukryte stany wszystkich warstw. Zobacz `hidden_states` w zwróconych tensorach, aby uzyskać
            więcej szczegółów.
        return_dict (`bool`, *optional*):
            Czy zwracać [`~utils.ModelOutput`] zamiast zwykłej krotki.
"""


# Rozszerzona klasa BlackholeEmbeddings do obsługi zamrażania warstw.
# Ta klasa będzie używać Twojej oryginalnej implementacji jako bazę.
class BlackholeEmbeddings(OriginalBlackholeEmbeddings):
    def __init__(self, config: BlackholeConfig):
        super().__init__(config)
        self.config = config

        # Implementacja logiki zamrażania ciężkich cech numerycznych
        # Zakładamy, że w Twoim OriginalBlackholeEmbeddings (hugging_embedding.py)
        # istnieje atrybut `numeric_embedding_projection`
        if config.numeric_heavy_feature_freeze:
            logger.info("Zamrażanie warstw 'heavy' cech numerycznych w BlackholeEmbeddings.")
            if hasattr(self, 'numeric_embedding_projection') and self.numeric_embedding_projection is not None:
                for param in self.numeric_embedding_projection.parameters():
                    param.requires_grad = False
                logger.info("Parametry numeric_embedding_projection zostały zamrożone.")
            else:
                logger.warning(
                    "Ostrzeżenie: `config.numeric_heavy_feature_freeze` jest `True`, "
                    "ale nie znaleziono atrybutu `numeric_embedding_projection` w `BlackholeEmbeddings`. "
                    "Upewnij się, że warstwy numeryczne są poprawnie zdefiniowane "
                    "i nazwane w `blackhole/embadding_hugging_face/hugging_embedding.py`."
                )

# --- Bazowy Model Blackhole (Encoder) ---

@add_start_docstrings(
    "The bare Blackhole Model transformer outputting raw hidden-states without any specific head on top.",
    "BlackholeConfig",
)
class BlackholeModel(PreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) or a decoder, in which case
    a causality mask will be added by default.
    """

    config_class = BlackholeConfig
    base_model_prefix = "blackhole"
    supports_gradient_checkpointing = True

    def __init__(self, config: BlackholeConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        # Używamy rozszerzonych embeddingów z logiką zamrażania
        self.embeddings = BlackholeEmbeddings(config)

        # Używamy standardowego BertEncoder, ponieważ Nova ma być architekturą Transformer
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Zwraca warstwę osadzającą tokeny wejściowe.
        Wymagane przez Hugging Face do prawidłowego łączenia wag w modelach MLM.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """
        Ustawia nową warstwę osadzającą tokeny wejściowe.
        Wymagane przez Hugging Face do prawidłowego łączenia wag w modelach MLM.
        """
        self.embeddings.word_embeddings = new_embeddings

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

    @add_start_docstrings_to_model_forward(BLACKHOLE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=BlackholeConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        numeric_values: Optional[torch.Tensor] = None, # Twoje dodatkowe wejście
        numeric_formats: Optional[torch.Tensor] = None, # Twoje dodatkowe wejście
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from blackhole.nova_hugging_face import BlackholeTokenizer, BlackholeModel, BlackholeConfig
        >>> import torch

        >>> # Inicjalizacja tokenizatora i konfiguracji (zakładając, że masz je już zapisane)
        >>> # Dla celów przykładu, podajemy minimalne parametry
        >>> tokenizer_path = "./blackhole_tokenizer_demo" # Zastąp ścieżką do Twojego zapisanego tokenizatora
        >>> tokenizer = BlackholeTokenizer.from_pretrained(tokenizer_path)
        >>> num_token_id = tokenizer.convert_tokens_to_ids(tokenizer.num_token)

        >>> config = BlackholeConfig(
        >>>    vocab_size=tokenizer.vocab_size,
        >>>    hidden_size=256,
        >>>    num_hidden_layers=2,
        >>>    num_attention_heads=4,
        >>>    intermediate_size=512,
        >>>    max_position_embeddings=128,
        >>>    pad_token_id=tokenizer.pad_token_id,
        >>>    num_token_id=num_token_id,
        >>>    numeric_feature_dims={
        >>>        "float64_binary_repr": 64, "digit_pos_0": 10, "digit_pos_1": 10,
        >>>        "log_value": 1, "sign": 1, "exponent_base10": 1,
        >>>        "num_total_digits": 1, "num_decimal_places": 1,
        >>>        "is_integer_flag": 1, "is_positive_flag": 1, "is_zero_flag": 1,
        >>>        "is_negative_flag": 1, "is_power_of_2_flag": 1,
        >>>        "format_type_int": 1, "format_type_float": 1,
        >>>    },
        >>> )

        >>> model = BlackholeModel(config)

        >>> sentence = "The price is [NUM] dollars."
        >>> inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=config.max_position_embeddings)
        >>>
        >>> outputs = model(
        >>>    input_ids=inputs["input_ids"],
        >>>    attention_mask=inputs["attention_mask"],
        >>>    numeric_values=inputs["numeric_values"],
        >>>    numeric_formats=inputs["numeric_formats"]
        >>> )
        >>>
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output
        >>>
        >>> print(f"Kształt ostatniej ukrytej warstwy: {last_hidden_state.shape}")
        >>> print(f"Kształt wyjścia poolera: {pooled_output.shape}")
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Nie możesz jednocześnie określić input_ids i inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Musisz określić input_ids lub inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # Możemy dostarczyć maskę self-attention o wymiarach [batch_size, from_seq_length, to_seq_length]
        # sami, w takim przypadku wystarczy, że będzie ona rozgłaszalna do wszystkich głów.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Przygotuj maskę głowy, jeśli potrzebna
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Przekazujemy wszystkie niezbędne argumenty do embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            numeric_values=numeric_values,
            numeric_formats=numeric_formats,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# --- Model Blackhole z głową do Masked Language Modeling (MLM) ---

@add_start_docstrings(
    """
    Blackhole Model with a `language modeling` head on top (for masked language modeling).
    """,
    "BlackholeConfig",
)
class BlackholeForMaskedLM(PreTrainedModel):
    config_class = BlackholeConfig
    base_model_prefix = "blackhole"
    _keys_to_ignore_on_load_missing = [r"position_ids", r"cls.decoder.bias"]

    def __init__(self, config: BlackholeConfig):
        super().__init__(config)
        self.config = config

        self.blackhole = BlackholeModel(config, add_pooling_layer=False)
        self.cls = BertLMPredictionHead(config)

        self.cls.decoder.weight = self.blackhole.embeddings.word_embeddings.weight
        self.cls.bias = self.cls.decoder.bias

        self.post_init()

    def get_output_embeddings(self):
        # POPRAWKA TUTAJ: BertLMPredictionHead bezpośrednio posiada atrybut 'decoder'
        return self.cls.decoder

    def set_output_embeddings(self, new_embeddings):
        # POPRAWKA TUTAJ: BertLMPredictionHead bezpośrednio posiada atrybut 'decoder'
        self.cls.decoder = new_embeddings
        if hasattr(new_embeddings, 'bias') and new_embeddings.bias is not None:
             self.cls.bias = new_embeddings.bias
        else:
            self.cls.bias = nn.Parameter(torch.zeros(new_embeddings.weight.shape[0]))


    @add_start_docstrings_to_model_forward(BLACKHOLE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=BlackholeConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        numeric_values: Optional[torch.Tensor] = None,
        numeric_formats: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Etykiety do obliczania straty zamaskowanego modelowania języka.
            Indeksy powinny być w `[-100, 0, ..., config.vocab_size]` (zobacz docstring `input_ids`)
            Tokeny z indeksami ustawionymi na `-100` są ignorowane (maskowane), strata jest obliczana tylko dla zamaskowanych tokenów.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.blackhole(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            numeric_values=numeric_values,
            numeric_formats=numeric_formats,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertLMPredictionHead(nn.Module):
    def __init__(self, config: BlackholeConfig):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # Wagi wyjściowe są takie same jak osadzenia wejściowe, ale istnieje
        # bias wyjściowy dla każdego tokena.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Potrzebne jest połączenie między dwiema zmiennymi, aby bias był prawidłowo zmieniany za pomocą `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config: BlackholeConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

# --- Model Blackhole z głową do Klasyfikacji Sekwencji ---

@add_start_docstrings(
    """
    Blackhole Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """,
    "BlackholeConfig",
)
class BlackholeForSequenceClassification(PreTrainedModel):
    config_class = BlackholeConfig
    base_model_prefix = "blackhole"

    def __init__(self, config: BlackholeConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.blackhole = BlackholeModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BLACKHOLE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=BlackholeConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        numeric_values: Optional[torch.Tensor] = None,
        numeric_formats: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Etykiety do obliczania straty klasyfikacji/regresji sekwencji.
            Indeksy powinny być w `[0, ..., config.num_labels - 1]`. Jeśli `config.num_labels == 1`
            obliczana jest strata regresji (błąd średniokwadratowy), jeśli `config.num_labels > 1`
            obliczana jest strata klasyfikacji (entropia krzyżowa).

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.blackhole(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            numeric_values=numeric_values,
            numeric_formats=numeric_formats,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

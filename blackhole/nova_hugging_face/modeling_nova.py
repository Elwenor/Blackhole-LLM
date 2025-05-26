import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers.activations import ACT2FN
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
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded input. This is useful
            if you want more control over how your input embeddings are generated.

            If `inputs_embeds` is passed, then `input_ids` and `token_type_ids` (if specified) are ignored.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Rozszerzona klasa BlackholeEmbeddings do obsługi zamrażania warstw.
# Ta klasa będzie używać Twojej oryginalnej implementacji jako bazę.
class BlackholeEmbeddings(OriginalBlackholeEmbeddings):
    def __init__(self, config: BlackholeConfig):
        super().__init__(config)
        self.config = config

        # Implementacja logiki zamrażania ciężkich cech numerycznych
        # Zakładamy, że w Twoim OriginalBlackholeEmbeddings (hugging_embedding.py)
        # istnieje atrybut `heavy_numeric_projection` (lub podobny)
        # który odpowiada za te cechy. Jeśli nazwa jest inna, należy ją tu dostosować.
        if config.numeric_heavy_feature_freeze:
            logger.info("Zamrażanie warstw 'heavy' cech numerycznych w BlackholeEmbeddings.")
            # Poprawione: Sprawdzamy czy atrybut numeric_embedding_projection istnieje
            # Zgodnie z Twoim plikiem `hugging_embedding.py`, ta warstwa nazywa się `numeric_embedding_projection`
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
    "BlackholeConfig", # Changed from BlackholeConfig to "BlackholeConfig"
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

    # >>> DODAJ TEN FRAGMENT KODU <<<
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
    # >>> KONIEC FRAGMENTU KODU <<<

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
        >>>     vocab_size=tokenizer.vocab_size,
        >>>     hidden_size=256,
        >>>     num_hidden_layers=2,
        >>>     num_attention_heads=4,
        >>>     intermediate_size=512,
        >>>     max_position_embeddings=128,
        >>>     pad_token_id=tokenizer.pad_token_id,
        >>>     num_token_id=num_token_id,
        >>>     numeric_feature_dims={
        >>>         "float64_binary_repr": 64, "digit_pos_0": 10, "digit_pos_1": 10,
        >>>         "log_value": 1, "sign": 1, "exponent_base10": 1,
        >>>         "num_total_digits": 1, "num_decimal_places": 1,
        >>>         "is_integer_flag": 1, "is_positive_flag": 1, "is_zero_flag": 1,
        >>>         "is_negative_flag": 1, "is_power_of_2_flag": 1,
        >>>         "format_type_int": 1, "format_type_float": 1,
        >>>     },
        >>> )

        >>> model = BlackholeModel(config)

        >>> sentence = "The price is [NUM] dollars."
        >>> inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=config.max_position_embeddings)
        >>>
        >>> outputs = model(
        >>>     input_ids=inputs["input_ids"],
        >>>     attention_mask=inputs["attention_mask"],
        >>>     numeric_values=inputs["numeric_values"],
        >>>     numeric_formats=inputs["numeric_formats"]
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
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
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
    "BlackholeConfig", # Changed from BlackholeConfig to "BlackholeConfig"
)
class BlackholeForMaskedLM(PreTrainedModel):
    config_class = BlackholeConfig
    base_model_prefix = "blackhole"
    # Zmieniono klucz do ignorowania - 'predictions' już nie jest zagnieżdżone
    _keys_to_ignore_on_load_missing = [r"position_ids", r"cls.decoder.bias"]

    def __init__(self, config: BlackholeConfig):
        super().__init__(config)
        self.config = config

        self.blackhole = BlackholeModel(config, add_pooling_layer=False)
        # BertLMPredictionHead już zawiera w sobie transformację i dekoder
        self.cls = BertLMPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        # POPRAWKA TUTAJ: BertLMPredictionHead bezpośrednio posiada atrybut 'decoder'
        return self.cls.decoder

    def set_output_embeddings(self, new_embeddings):
        # POPRAWKA TUTAJ: BertLMPredictionHead bezpośrednio posiada atrybut 'decoder'
        self.cls.decoder = new_embeddings

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
            Labels for computing the masked language modeling loss.
            Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring)
            Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the masked tokens.

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

        # The output weights are the same as the input embeddings, but there is
        # an output bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
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
    "BlackholeConfig", # Changed from BlackholeConfig to "BlackholeConfig"
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
            Labels for computing the sequence classification/regression loss.
            Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1`
            a regression loss is computed (Mean-Square Error), If `config.num_labels > 1`
            a classification loss is computed (Cross-Entropy).

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
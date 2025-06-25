# nova_test_file/modeling_nova.py

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
from transformers.models.bert.modeling_bert import (
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

# *** CRUCIAL CHANGE HERE: IMPORTING BlackholeEmbeddings RELATIVELY ***
# Assuming you have moved the content of 'blackhole/embadding_hugging_face/hugging_embedding.py'
# into 'nova_test_file/hugging_embedding.py'
from .hugging_embedding import BlackholeEmbeddings as OriginalBlackholeEmbeddings

logger = logging.get_logger(__name__)


# DOCSTRING
BLACKHOLE_INPUT = """
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded input. This is useful if
            you want more control over how your input embeddings are generated.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~transformers.file_utils.ModelOutput`] instead of a plain tuple.
"""

class BlackholeEmbeddings(OriginalBlackholeEmbeddings):
    """
    Custom Blackhole Embeddings class, inheriting from your original BlackholeEmbeddings.
    This allows for potential extensions or overrides while keeping the core logic.
    """
    pass

class BlackholeModel(PreTrainedModel):
    """
    The bare Blackhole Model transformer outputting raw hidden-states without any specific head on top.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to training and
    inference.

    Parameters:
        config ([`BlackholeConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the documentation for more details.
    """
    config_class = BlackholeConfig
    base_model_prefix = "blackhole"

    def __init__(self, config: BlackholeConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BlackholeEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final preparations
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See detailed explanation in [base class documentation](https://huggingface.co/docs/transformers/main/en/main_classes/modeling#transformers.PreTrainedModel.prune_heads)
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BLACKHOLE_INPUT.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class="AutoTokenizer",
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class="AutoConfig",
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Dodane wejścia dla cech numerycznych
        numeric_values: Optional[torch.Tensor] = None,
        numeric_formats: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
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

        # We need to make sure the attention mask and head mask are of the correct shape
        extended_attention_mask: torch.Tensor = self.encoder.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # Prepare head mask if needed
        head_mask = self.encoder.get_head_mask(head_mask, self.config.num_hidden_layers)


        # Embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            numeric_values=numeric_values,
            numeric_formats=numeric_formats
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

class BlackholeForMaskedLM(PreTrainedModel):
    config_class = BlackholeConfig
    base_model_prefix = "blackhole"

    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.blackhole = BlackholeModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final preparations
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BLACKHOLE_INPUT.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class="AutoTokenizer",
        checkpoint="bert-base-uncased",
        output_type=MaskedLMOutput,
        config_class="AutoConfig",
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Dodane wejścia dla cech numerycznych
        numeric_values: Optional[torch.Tensor] = None,
        numeric_formats: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.blackhole(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            numeric_values=numeric_values,
            numeric_formats=numeric_formats
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def _tie_weights(self):
        # Make sure this module's decoder weights are tied with its input embeddings.
        # This is not strictly necessary but it makes the `BertLMPredictionHead`
        # easier to configure and thus `PreTrainedModel.resize_token_embeddings` works
        # correctly.
        self.bias = self.decoder.bias

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BlackholeForSequenceClassification(PreTrainedModel):
    config_class = BlackholeConfig
    base_model_prefix = "blackhole"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.blackhole = BlackholeModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final preparations
        self.post_init()

    @add_start_docstrings_to_model_forward(BLACKHOLE_INPUT.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class="AutoTokenizer",
        checkpoint="bert-base-uncased",
        output_type=SequenceClassifierOutput,
        config_class="AutoConfig",
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Dodane wejścia dla cech numerycznych
        numeric_values: Optional[torch.Tensor] = None,
        numeric_formats: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.blackhole(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            numeric_values=numeric_values,
            numeric_formats=numeric_formats,
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
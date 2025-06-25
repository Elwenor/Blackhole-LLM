# scripts/training/model.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional

# Corrected: Import BlackholeConfig from the new nova_test_file location
from nova_test_file.configuration_nova import BlackholeConfig

class BlackholeSeq2SeqForConditionalGeneration(PreTrainedModel):
    """
    A Sequence-to-Sequence model for conditional generation that handles both
    textual and numerical features.
    """
    config_class = BlackholeConfig
    base_model_prefix = "blackhole_seq2seq"

    def __init__(self, config: BlackholeConfig):
        super().__init__(config)
        self.config = config

        # --- Embeddings ---
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # Projects the raw numeric features into the model's hidden dimension.
        # WAŻNE: config.numeric_input_features = 7 (z config.py)
        # Właśnie dlatego łączymy numeric_values (1) i numeric_formats (6) w data_processing.py
        self.numeric_embedding = nn.Linear(config.numeric_input_features, config.hidden_size)

        # --- Transformer Blocks ---
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads, dim_feedforward=config.intermediate_size, dropout=config.dropout, batch_first=True, activation='gelu'),
            num_layers=config.encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads, dim_feedforward=config.intermediate_size, dropout=config.dropout, batch_first=True, activation='gelu'),
            num_layers=config.decoder_layers
        )

        # --- Prediction Heads ---
        # For token prediction
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        # For numeric feature prediction
        self.numeric_head = nn.Linear(config.hidden_size, config.numeric_input_features)

        # Initialize weights
        self.post_init()

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": total_params, "trainable_params": trainable_params}

    def _prepare_decoder_attention_mask(self, tgt_len: int, device: torch.device):
        """Generates a causal mask for decoder self-attention."""
        mask = torch.full((tgt_len, tgt_len), float('-inf'), device=device, dtype=torch.float)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_numeric_values: torch.Tensor,
        encoder_numeric_formats: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        # Zmienione: Przyjmujemy oddzielnie values i formats dla dekodera
        decoder_numeric_values: Optional[torch.Tensor] = None,
        decoder_numeric_formats: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Seq2SeqLMOutput:
        # Połącz embeddingi tokenów i cech numerycznych dla enkodera
        encoder_numeric_features_combined = torch.cat(
            (encoder_numeric_values.unsqueeze(-1), encoder_numeric_formats), dim=-1
        )
        encoder_numeric_embeds = self.numeric_embedding(encoder_numeric_features_combined)
        encoder_token_embeds = self.token_embedding(encoder_input_ids)
        encoder_input_embeds = encoder_token_embeds + encoder_numeric_embeds

        # Encoder forward pass
        encoder_outputs = self.encoder(
            src=encoder_input_embeds,
            src_key_padding_mask=(encoder_attention_mask == 0) # Maska paddingu dla enkodera
        )

        if decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels) if labels is not None else None

        # Połącz embeddingi tokenów i cech numerycznych dla dekodera
        decoder_token_embeds = self.token_embedding(decoder_input_ids)

        # Nowe: Połącz decoder_numeric_values i decoder_numeric_formats
        if decoder_numeric_values is not None and decoder_numeric_formats is not None:
            # Upewnij się, że rozmiary są zgodne przed konkatenacją
            # Sprawdzamy, czy tensory nie są puste ([0, ...])
            if decoder_numeric_values.numel() > 0 and decoder_numeric_formats.numel() > 0:
                decoder_numeric_features_combined = torch.cat(
                    (decoder_numeric_values.unsqueeze(-1), decoder_numeric_formats), dim=-1
                )
                decoder_numeric_embeds = self.numeric_embedding(decoder_numeric_features_combined)
                decoder_input_embeds = decoder_token_embeds + decoder_numeric_embeds
            else:
                # Jeśli któryś z tensorów jest pusty, używamy tylko token_embeds
                decoder_input_embeds = decoder_token_embeds
        else:
            decoder_input_embeds = decoder_token_embeds


        # Decoder attention mask (causal mask for self-attention)
        tgt_len = decoder_input_ids.shape[1]
        causal_mask = self._prepare_decoder_attention_mask(tgt_len, decoder_input_ids.device)

        # Decoder padding mask
        decoder_padding_mask = (decoder_input_ids == self.config.pad_token_id)

        # Decoder forward pass
        decoder_output = self.decoder(
            tgt=decoder_input_embeds,
            memory=encoder_outputs,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_padding_mask,
            memory_key_padding_mask=(encoder_attention_mask == 0)
        )

        # --- Prediction Heads ---
        token_logits = self.lm_head(decoder_output)
        predicted_numeric_features = self.numeric_head(decoder_output)

        # --- Loss Calculation ---
        loss = None
        # 1. Token Prediction Loss (Cross-Entropy)
        loss_fct_token = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        token_loss = loss_fct_token(token_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 2. Numeric Feature Prediction Loss (MSE)
        numeric_loss = torch.tensor(0.0, device=self.device) # Domyślnie 0
        if decoder_numeric_values is not None and decoder_numeric_formats is not None:
            num_token_mask = (labels == self.config.num_token_id)
            if num_token_mask.sum() > 0:
                loss_fct_numeric = nn.MSELoss()
                # Łączymy "ground truth" numeryczne cechy w taki sam sposób, jak w przypadku inputów
                # aby dopasować kształt do predicted_numeric_features
                actual_numeric_features_combined = torch.cat(
                    (decoder_numeric_values.unsqueeze(-1), decoder_numeric_formats), dim=-1
                )
                numeric_loss = loss_fct_numeric(
                    predicted_numeric_features[num_token_mask],
                    actual_numeric_features_combined[num_token_mask]
                )

        # Combine losses (e.g., weighted sum)
        loss = token_loss + numeric_loss # You might want to add weights here if needed

        return Seq2SeqLMOutput(
            loss=loss,
            logits=token_logits,
            decoder_hidden_states=decoder_output,
            encoder_last_hidden_state=encoder_outputs,
        )

    def _shift_right(self, input_ids: torch.Tensor):
        # Shift input_ids to the right to create decoder_input_ids
        # (remove last token, add bos_token_id at the beginning)
        decoder_input_ids = input_ids.clone()
        if len(decoder_input_ids.shape) == 1: # Handle single sequence
            decoder_input_ids = decoder_input_ids.unsqueeze(0)

        # Create a tensor with bos_token_id
        bos_tensor = torch.full(
            (decoder_input_ids.shape[0], 1), self.config.bos_token_id,
            dtype=decoder_input_ids.dtype, device=decoder_input_ids.device
        )

        # Concatenate bos_token_id and all but the last token of input_ids
        # This handles the case where input_ids is already max_length
        shifted_input_ids = torch.cat([bos_tensor, decoder_input_ids[:, :-1]], dim=-1)
        return shifted_input_ids.squeeze() if len(decoder_input_ids.shape) == 1 else shifted_input_ids
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union, Dict, List, Any
import collections.abc
import logging

logger = logging.getLogger(__name__)

class BlackholeDataCollatorForSeq2Seq:
    """
    Data Collator for Seq2Seq models.
    Pads and batches all inputs for both encoder and decoder, including custom numeric features.
    Ensures keys match the model's forward() signature.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, model=None, padding=True,
                 max_length: Optional[int] = None, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        # Użyj max_length z konfiguracji lub tokenizer.model_max_length
        self.max_length = max_length if max_length is not None else tokenizer.model_max_length
        self.label_pad_token_id = label_pad_token_id
        # Użyj NaN dla paddingu numerycznego, co jest często bezpieczne dla floatów
        self.numeric_padding_value = float('nan') 
        
        # Upewnij się, że numeric_feature_size jest ustawiony, w przeciwnym razie będzie 0
        self._numeric_feature_size = getattr(tokenizer, '_numeric_feature_size', 0)
        if self._numeric_feature_size == 0:
            logger.warning("Tokenizer's _numeric_feature_size is 0. Numeric features might not be processed correctly.")


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # --- Instrukcje debugowania DataCollator ---
        print(f"DEBUG: Loading from data_collator2.py: {__file__}")
        print(f"DEBUG: DataCollator received a batch with {len(features)} features.")
        if len(features) > 0:
            print(f"DEBUG: First feature keys in collator: {list(features[0].keys())}")
            for i, f_item in enumerate(features[:min(len(features), 3)]):
                print(f"DEBUG:   Feature {i} keys: {list(f_item.keys())}")
                for k, v in f_item.items():
                    if isinstance(v, torch.Tensor):
                        print(f"DEBUG:     {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"DEBUG:     {k}: type={type(v)}")
        # --- Koniec instrukcji debugowania DataCollator ---

        # Separate inputs for encoder and decoder
        # Encoder inputs:
        encoder_input_ids = [f["encoder_input_ids"] for f in features]
        encoder_attention_mask = [f["encoder_attention_mask"] for f in features]
        encoder_numeric_values = [f["encoder_numeric_values"] for f in features]
        encoder_numeric_formats = [f["encoder_numeric_formats"] for f in features]

        # Decoder inputs (labels and associated numeric features):
        labels = [f["labels"] for f in features]
        decoder_numeric_values = [f["decoder_numeric_values"] for f in features]
        decoder_numeric_formats = [f["decoder_numeric_formats"] for f in features]


        # Determine max sequence length for encoder inputs
        max_encoder_len = max(len(t) for t in encoder_input_ids) if encoder_input_ids else 0
        if self.max_length and max_encoder_len > self.max_length:
            max_encoder_len = self.max_length

        # Determine max sequence length for decoder labels
        max_label_len = max(len(t) for t in labels) if labels else 0
        if self.max_length and max_label_len > self.max_length:
            max_label_len = self.max_length


        # Pad and stack ENCODER inputs
        # input_ids
        padded_encoder_input_ids = [
            torch.cat([t, torch.full((max_encoder_len - len(t),), self.tokenizer.pad_token_id, dtype=torch.long)])
            for t in encoder_input_ids
        ]
        batch_encoder_input_ids = torch.stack(padded_encoder_input_ids)

        # attention_mask
        padded_encoder_attention_mask = [
            torch.cat([t, torch.full((max_encoder_len - len(t),), 0, dtype=torch.long)])
            for t in encoder_attention_mask
        ]
        batch_encoder_attention_mask = torch.stack(padded_encoder_attention_mask)

        # numeric_values
        padded_encoder_numeric_values = []
        for t in encoder_numeric_values:
            padding_len = max_encoder_len - t.shape[0]
            if padding_len > 0:
                padded_encoder_numeric_values.append(torch.cat([t, torch.full((padding_len,), self.numeric_padding_value, dtype=torch.float32)]))
            else:
                padded_encoder_numeric_values.append(t)
        batch_encoder_numeric_values = torch.stack(padded_encoder_numeric_values)
        
        # numeric_formats (assuming shape [seq_len, numeric_feature_dim])
        padded_encoder_numeric_formats = []
        for t in encoder_numeric_formats:
            padding_len = max_encoder_len - t.shape[0]
            if padding_len > 0:
                padding_tensor = torch.full((padding_len, self._numeric_feature_size), self.numeric_padding_value, dtype=torch.float32)
                if t.numel() > 0:
                    padded_encoder_numeric_formats.append(torch.cat([t, padding_tensor], dim=0))
                else:
                    padded_encoder_numeric_formats.append(padding_tensor)
            else:
                padded_encoder_numeric_formats.append(t)
        batch_encoder_numeric_formats = torch.stack(padded_encoder_numeric_formats)


        # Pad and stack DECODER inputs (labels and decoder_numeric_features)
        # labels
        padded_labels = [
            torch.cat([t, torch.full((max_label_len - len(t),), self.label_pad_token_id, dtype=torch.long)])
            for t in labels
        ]
        batch_labels = torch.stack(padded_labels)

        # decoder_numeric_values
        padded_decoder_numeric_values = []
        for t in decoder_numeric_values:
            padding_len = max_label_len - t.shape[0]
            if padding_len > 0:
                padded_decoder_numeric_values.append(torch.cat([t, torch.full((padding_len,), self.numeric_padding_value, dtype=torch.float32)]))
            else:
                padded_decoder_numeric_values.append(t)
        batch_decoder_numeric_values = torch.stack(padded_decoder_numeric_values)

        # decoder_numeric_formats
        padded_decoder_numeric_formats = []
        for t in decoder_numeric_formats:
            padding_len = max_label_len - t.shape[0]
            if padding_len > 0:
                padding_tensor = torch.full((padding_len, self._numeric_feature_size), self.numeric_padding_value, dtype=torch.float32)
                if t.numel() > 0:
                    padded_decoder_numeric_formats.append(torch.cat([t, padding_tensor], dim=0))
                else:
                    padded_decoder_numeric_formats.append(padding_tensor)
            else:
                padded_decoder_numeric_formats.append(t)
        batch_decoder_numeric_formats = torch.stack(padded_decoder_numeric_formats)


        # Assemble the final batch dictionary, matching model.forward() arguments
        batch = {
            "encoder_input_ids": batch_encoder_input_ids,
            "encoder_attention_mask": batch_encoder_attention_mask,
            "encoder_numeric_values": batch_encoder_numeric_values,
            "encoder_numeric_formats": batch_encoder_numeric_formats,
            "labels": batch_labels,
            "decoder_numeric_values": batch_decoder_numeric_values,
            "decoder_numeric_formats": batch_decoder_numeric_formats,
            # Trainer will automatically create decoder_input_ids from labels internally for training
        }
        
        return batch
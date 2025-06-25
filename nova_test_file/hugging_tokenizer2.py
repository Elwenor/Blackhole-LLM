# blackhole/tokenizer_hugging_face/hugging_tokenizer2.py

import os
import collections
import re
import unicodedata
import json
import torch
import math
from transformers import PreTrainedTokenizerFast, BatchEncoding
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, Regex
from tokenizers.processors import TemplateProcessing
from tokenizers import Encoding as TokenizersEncoding
from typing import List, Dict, Optional, Tuple, Union, Any, Iterator
import logging

logger = logging.getLogger(__name__)

# Definiowanie tokenów specjalnych
NUMBER_TOKEN = "[NUM]"
CAPITALIZED_TOKEN = "[CAP]"
ALL_CAPS_TOKEN = "[ALLCAPS]"
SPACE_TOKEN = "[SPACE]" # Dodany nowy token specjalny (jeśli go używasz)

CUSTOM_SPECIAL_TOKENS = {
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "mask_token": "[MASK]",
    "number_token": NUMBER_TOKEN,
    "capitalized_token": CAPITALIZED_TOKEN,
    "all_caps_token": ALL_CAPS_TOKEN,
    "space_token": SPACE_TOKEN, # Dodany nowy token specjalny (jeśli go używasz)
}

class BlackholeTokenizer2(PreTrainedTokenizerFast):
    vocab_files_names = {"vocab_file": "vocab.json", "tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask", "numeric_values", "numeric_formats"]

    num_token = NUMBER_TOKEN
    cap_token = CAPITALIZED_TOKEN
    allcaps_token = ALL_CAPS_TOKEN
    space_token = SPACE_TOKEN

    numeric_padding_value = 0.0 # Zmieniono z float('nan') na 0.0 dla spójności i kompatybilności z PyTorch.

    _numeric_feature_size: int = None # Domyślna wartość, zostanie ustawiona w configu lub z tokenizatora

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        num_token: str = NUMBER_TOKEN,
        cap_token: str = CAPITALIZED_TOKEN,
        allcaps_token: str = ALL_CAPS_TOKEN,
        space_token: str = SPACE_TOKEN,
        numeric_feature_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Initializes a BlackholeTokenizer.

        Args:
            vocab_file (`str`): Path to the vocabulary file.
            tokenizer_file (`str`, *optional*): Path to the tokenizer file.
            numeric_feature_size (`int`, *optional*): The expected dimension of numeric features.
                                                     If None, it must be set later.
            **kwargs: Arguments for the base PreTrainedTokenizerFast class.
        """
        if tokenizer_file is not None:
            tokenizer = Tokenizer.from_file(tokenizer_file)
        elif vocab_file is not None:
            tokenizer = self._build_default_tokenizer_from_vocab(vocab_file)
        else:
            # This case is usually for from_pretrained without a local file,
            # so the tokenizer object will be built by super().from_pretrained
            tokenizer = None

        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            # Add custom tokens to special_tokens_map
            additional_special_tokens=[num_token, cap_token, allcaps_token, space_token],
            **kwargs,
        )

        self.num_token = num_token
        self.cap_token = cap_token
        self.allcaps_token = allcaps_token
        self.space_token = space_token

        # Set _numeric_feature_size
        if numeric_feature_size is not None:
            self._numeric_feature_size = numeric_feature_size
        elif hasattr(self, 'init_kwargs') and 'numeric_feature_size' in self.init_kwargs:
            self._numeric_feature_size = self.init_kwargs['numeric_feature_size']
        elif hasattr(self, 'config') and hasattr(self.config, 'numeric_input_features'):
             self._numeric_feature_size = self.config.numeric_input_features
        else:
            logger.warning(f"Numeric feature size not set during BlackholeTokenizer initialization or from config. "
                           f"This must be set manually via tokenizer._numeric_feature_size = value "
                           f"or passed during initialization for proper numeric feature handling.")
            # Default to a value, based on your config.py value (7)
            self._numeric_feature_size = 7 


    def _build_default_tokenizer_from_vocab(self, vocab_file):
        """Builds a default Tokenizer object from a vocab.json file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token=self.unk_token))

        # Define pre-tokenizer to split text into words
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Split(Regex(r"[0-9]+(?:[\.,][0-9]+)?"), "isolated"), # Split numbers
            pre_tokenizers.Split(Regex(r"[^\w\s]"), "isolated"), # Split punctuation
        ])

        # Add custom special tokens to the tokenizer's vocabulary
        special_tokens_list = list(CUSTOM_SPECIAL_TOKENS.values())
        tokenizer.add_special_tokens(special_tokens_list)

        # Post-processor to handle special tokens for generation
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.cls_token_id),
                ("[SEP]", self.sep_token_id),
            ],
        )
        tokenizer.decoder = decoders.WordPiece()
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def _tokenize(self, text: str) -> List[str]:
        """Converts a string in a sequence of tokens (string)."""
        # For a Fast Tokenizer, `encode` or `_call_backend_tokenizer` is usually used directly.
        # This method is primarily for compatibility with older PreTrainedTokenizer.
        # The actual tokenization for fast tokenizers happens in `tokenizer.encode()`.
        
        # A basic implementation just to satisfy the abstract method
        # It's better to rely on `self.tokenizer.encode` for actual tokenization.
        return self.tokenizer.encode(text, add_special_tokens=False).tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.id_to_token(index)

    def _encode_text_and_numeric(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        return_tensors: Optional[str] = None,
        return_numeric_features: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Encodes text, extracts numeric features, and returns a BatchEncoding.
        """
        # Encode the text using the internal Tokenizer
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            # Padding is handled later to apply numeric_padding_value
        )

        input_ids = torch.tensor(encoded.ids, dtype=torch.long)
        attention_mask = torch.tensor(encoded.attention_mask, dtype=torch.long)

        numeric_values = torch.full((len(encoded.ids),), self.numeric_padding_value, dtype=torch.float32)
        numeric_formats = torch.full((len(encoded.ids), self._numeric_feature_size), self.numeric_padding_value, dtype=torch.float32)

        if return_numeric_features and self._numeric_feature_size is not None:
            # Extract numeric features and formats
            for i, (token_id, token_str) in enumerate(zip(encoded.ids, encoded.tokens)):
                if token_id == self.num_token_id:
                    # Look for the actual number string from the original text if possible,
                    # by finding the span of the token in the original text.
                    span = encoded.token_to_original_range(i)
                    original_word = None
                    if span:
                        original_word = text[span[0]:span[1]]
                    
                    if original_word and re.match(r'^-?\d+(\.\d+)?$', original_word):
                        try:
                            # Use your number_embedding_features function
                            features = self._number_embedding_features(float(original_word))
                            numeric_values[i] = float(original_word)
                            numeric_formats[i, :len(features)] = torch.tensor(features, dtype=torch.float32)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not convert '{original_word}' to number or generate features: {e}")
                    else:
                        logger.warning(f"Could not reliably extract original number for [NUM] token at index {i} ('{token_str}').")
                elif token_id in [self.cap_token_id, self.allcaps_token_id, self.space_token_id]:
                    # For [CAP], [ALLCAPS], [SPACE] tokens, their numeric features might be 0 or some placeholder
                    numeric_values[i] = 0.0 # Or some other specific value
                    numeric_formats[i, :] = torch.full((self._numeric_feature_size,), 0.0, dtype=torch.float32) # All zeros

        # Handle padding if requested (usually done by DataCollator in training loop)
        if padding == 'max_length' and max_length is not None:
            current_len = len(input_ids)
            if current_len < max_length:
                pad_len = max_length - current_len
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.full((pad_len,), 0, dtype=torch.long)])
                numeric_values = torch.cat([numeric_values, torch.full((pad_len,), self.numeric_padding_value, dtype=torch.float32)])
                numeric_formats = torch.cat([numeric_formats, torch.full((pad_len, self._numeric_feature_size), self.numeric_padding_value, dtype=torch.float32)])
            elif current_len > max_length and truncation is True: # Already truncated by tokenizer.encode, but for safety
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                numeric_values = numeric_values[:max_length]
                numeric_formats = numeric_formats[:max_length]

        # Apply desired tensor format
        if return_tensors == "pt":
            # For a single sequence, unsqueeze to add batch dimension
            input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            attention_mask = attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask
            numeric_values = numeric_values.unsqueeze(0) if numeric_values.dim() == 1 else numeric_values
            numeric_formats = numeric_formats.unsqueeze(0) if numeric_formats.dim() == 2 else numeric_formats

        batch_encoding = BatchEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "numeric_values": numeric_values,
            "numeric_formats": numeric_formats,
        })

        return batch_encoding

    def __call__(self, *args, return_numeric_features: bool = False, **kwargs) -> BatchEncoding:
        """
        Main method to tokenize and prepare inputs.
        Delegates to _encode_text_and_numeric for custom logic.
        Handles both single text and batch processing.
        """
        if isinstance(args[0], (list, tuple)): # It's a batch of texts
            batch_texts = args[0]
            # Process each text individually
            encoded_individual_inputs = [
                self._encode_text_and_numeric(
                    text,
                    add_special_tokens=kwargs.get("add_special_tokens", True),
                    max_length=kwargs.get("max_length", None),
                    padding=False, # Padding is handled by DataCollator in training, or explicitly here
                    truncation=kwargs.get("truncation", False),
                    return_tensors="pt", # Return PyTorch tensors for individual encoding
                    return_numeric_features=return_numeric_features,
                ) for text in batch_texts
            ]
            
            # Manually stack the tensors to form a batch
            # This is a simplified batching. In a real training loop, a DataCollator is used
            # to handle padding and stacking more robustly across the batch.
            batched_inputs = {}
            for key in ["input_ids", "attention_mask", "numeric_values", "numeric_formats"]:
                if encoded_individual_inputs[0].get(key) is not None:
                    batched_inputs[key] = torch.cat([e[key] for e in encoded_individual_inputs], dim=0)
            
            return BatchEncoding(batched_inputs)

        else: # Single text processing
            text = args[0]
            return self._encode_text_and_numeric(
                text,
                add_special_tokens=kwargs.get("add_special_tokens", True),
                max_length=kwargs.get("max_length", None),
                padding=kwargs.get("padding", False),
                truncation=kwargs.get("truncation", False),
                return_tensors=kwargs.get("return_tensors", None),
                return_numeric_features=return_numeric_features,
            )

    def _number_embedding_features(self, number: float) -> List[float]:
        """
        Generates a feature vector for a given number.
        This function should match the one used during pretraining if applicable.
        It should produce DETERMINE_NUMERIC_FEATURE_DIM (7) features.
        """
        features = []

        # 1. Value itself (no normalization, model can learn to scale)
        features.append(float(number)) 

        # 2. Log of absolute value (handle zero/negatives)
        features.append(math.log(abs(number)) if number != 0 else 0.0)

        # 3. Sign
        features.append(1.0 if number >= 0 else -1.0)

        # 4. Number of decimal places
        s = str(number)
        if '.' in s:
            features.append(float(len(s.split('.')[-1])))
        else:
            features.append(0.0)

        # 5. Is integer
        features.append(1.0 if number == int(number) else 0.0)

        # 6. Order of magnitude (e.g., floor(log10(abs(number))))
        features.append(math.floor(math.log10(abs(number))) if abs(number) >= 1 else 0.0)
        
        # 7. Reciprocal (handle division by zero)
        features.append(1.0 / number if number != 0 else 0.0)

        # Ensure the feature list is exactly _numeric_feature_size long
        # The config.py shows DETERMINE_NUMERIC_FEATURE_DIM = 7
        if self._numeric_feature_size is not None and len(features) < self._numeric_feature_size:
            features.extend([0.0] * (self._numeric_feature_size - len(features)))
        elif self._numeric_feature_size is not None and len(features) > self._numeric_feature_size:
            features = features[:self._numeric_feature_size]
        
        return features


    # Required properties for PreTrainedTokenizerFast
    @property
    def is_fast(self) -> bool:
        return True

    @property
    def num_special_tokens_to_add(self) -> int:
        # [CLS], [SEP]
        return 2

    # Override save_pretrained and from_pretrained to handle custom attributes
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the tokenizer and its custom numeric_feature_size.
        """
        super().save_pretrained(save_directory, **kwargs)
        # Save custom attributes to tokenizer_config.json
        tokenizer_config_path = os.path.join(save_directory, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)
        else:
            tokenizer_config = {}

        if self._numeric_feature_size is not None:
            tokenizer_config['numeric_feature_size'] = self._numeric_feature_size
            with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        else:
            logger.warning("Numeric feature size was not set, skipping saving it to tokenizer_config.json.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *init_inputs, **kwargs):
        """
        Load the tokenizer and its custom numeric_feature_size.
        """
        # Load the base tokenizer using the parent class method
        instance = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

        # Load custom attributes from tokenizer_config.json
        tokenizer_config_path = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            try:
                with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                    tokenizer_config = json.load(f)
                if 'numeric_feature_size' in tokenizer_config:
                    instance._numeric_feature_size = tokenizer_config['numeric_feature_size']
                    logger.info(f"Loaded numeric_feature_size: {instance._numeric_feature_size} from tokenizer_config.json")
            except Exception as e:
                logger.warning(f"Could not load numeric_feature_size from tokenizer_config.json: {e}")
        
        # Allow setting numeric_feature_size from kwargs during from_pretrained if not found in config
        if instance._numeric_feature_size is None and 'numeric_feature_size' in kwargs:
             instance._numeric_feature_size = kwargs['numeric_feature_size']
             logger.info(f"Loaded numeric_feature_size: {instance._numeric_feature_size} from kwargs.")
        elif instance._numeric_feature_size is None:
            logger.warning("numeric_feature_size not found in tokenizer_config.json or kwargs during from_pretrained. Setting to default (7).")
            instance._numeric_feature_size = 7 # Fallback default to match config.py

        return instance
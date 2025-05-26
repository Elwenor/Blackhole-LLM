import os
import collections
import re
import unicodedata
import json
import torch
from transformers import PreTrainedTokenizerFast, BatchEncoding
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, Regex
from tokenizers.processors import TemplateProcessing
from tokenizers import Encoding as TokenizersEncoding
from typing import List, Dict, Optional, Tuple, Union, Any, Iterator

# Definiowanie tokenów specjalnych
NUMBER_TOKEN = "[NUM]"
CAPITALIZED_TOKEN = "[CAP]"
ALL_CAPS_TOKEN = "[ALLCAPS]"

CUSTOM_SPECIAL_TOKENS = {
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "mask_token": "[MASK]",
    "number_token": NUMBER_TOKEN,
    "capitalized_token": CAPITALIZED_TOKEN,
    "all_caps_token": ALL_CAPS_TOKEN
}

class BlackholeTokenizer(PreTrainedTokenizerFast):
    vocab_files_names = {"vocab_file": "vocab.json", "tokenizer_file": "tokenizer.json"}
    # Added numeric_formats to model_input_names
    model_input_names = ["input_ids", "attention_mask", "numeric_values", "numeric_formats"] 

    num_token = NUMBER_TOKEN
    cap_token = CAPITALIZED_TOKEN
    allcaps_token = ALL_CAPS_TOKEN

    # Define padding values for numeric tensors
    numeric_padding_value = float('nan') # Using NaN for numeric_values padding
    numeric_format_padding_value = -1 # Using -1 for numeric_formats padding

    # Mapping for numeric format strings to integer IDs
    _numeric_format_to_id = {
        'integer': 0,
        'decimal_float': 1,
        'scientific_notation': 2,
        'hexadecimal': 3,
        'unknown': -1 # Fallback for unknown formats
    }

    def __init__(self, vocab_file=None, tokenizer_file=None, **kwargs):
        self.tokenizer = None

        if tokenizer_file is not None and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.tokenizer = Tokenizer(models.BPE())

            special_token_values = list(CUSTOM_SPECIAL_TOKENS.values())
            self.tokenizer.add_special_tokens(special_token_values)

            # Pre-tokenizer divides raw text into segments (words, numbers, punctuation, spaces)
            # The actual insertion of special marker tokens ([NUM], [CAP], [ALLCAPS])
            # will happen in _prepare_text_for_bpe_and_collect_metadata
            # Zmieniono: Dodano [-+]? na początku wyrażenia regularnego dla liczb
            tokenizers_regex_pattern = Regex(
                r"([-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|https?://\S+|www\.\S+|[A-Za-z_]+(?:['\-][A-Za-z_]+)*|[^\s\w\d]|\s+)"
            )
            self.tokenizer.pre_tokenizer = pre_tokenizers.Split(
                pattern=tokenizers_regex_pattern,
                behavior="isolated"
            )

            cls_id = self.tokenizer.token_to_id(CUSTOM_SPECIAL_TOKENS["cls_token"])
            sep_id = self.tokenizer.token_to_id(CUSTOM_SPECIAL_TOKENS["sep_token"])

            if cls_id is None or sep_id is None:
                missing_special_tokens_list = []
                if cls_id is None: missing_special_tokens_list.append(CUSTOM_SPECIAL_TOKENS["cls_token"])
                if sep_id is None: missing_special_tokens_list.append(CUSTOM_SPECIAL_TOKENS["sep_token"])
                if missing_special_tokens_list:
                    self.tokenizer.add_special_tokens(missing_special_tokens_list)
                    cls_id = self.tokenizer.token_to_id(CUSTOM_SPECIAL_TOKENS["cls_token"])
                    sep_id = self.tokenizer.token_to_id(CUSTOM_SPECIAL_TOKENS["sep_token"])
                if cls_id is None or sep_id is None:
                    raise ValueError(f"Special tokens CLS or SEP not found or could not be added to tokenizer vocabulary. This is critical.")

            # Post-processor adds CLS/SEP tokens around the sequence
            self.tokenizer.post_processor = TemplateProcessing(
                single=f"{CUSTOM_SPECIAL_TOKENS['cls_token']} $A {CUSTOM_SPECIAL_TOKENS['sep_token']}",
                pair=f"{CUSTOM_SPECIAL_TOKENS['cls_token']} $A {CUSTOM_SPECIAL_TOKENS['sep_token']} $B {CUSTOM_SPECIAL_TOKENS['sep_token']}",
                special_tokens=[
                    (CUSTOM_SPECIAL_TOKENS["cls_token"], cls_id),
                    (CUSTOM_SPECIAL_TOKENS["sep_token"], sep_id),
                ],
            )
            self.tokenizer.decoder = decoders.BPEDecoder()

        resolved_add_prefix_space = kwargs.pop('add_prefix_space', False)
        kwargs.pop('additional_special_tokens', None)
        kwargs.pop('unk_token', None)
        kwargs.pop('cls_token', None)
        kwargs.pop('sep_token', None)
        kwargs.pop('pad_token', None)
        kwargs.pop('mask_token', None)

        super().__init__(
            tokenizer_object=self.tokenizer,
            unk_token=CUSTOM_SPECIAL_TOKENS["unk_token"],
            cls_token=CUSTOM_SPECIAL_TOKENS["cls_token"],
            sep_token=CUSTOM_SPECIAL_TOKENS["sep_token"],
            pad_token=CUSTOM_SPECIAL_TOKENS["pad_token"],
            mask_token=CUSTOM_SPECIAL_TOKENS["mask_token"],
            additional_special_tokens=[
                CUSTOM_SPECIAL_TOKENS["number_token"],
                CUSTOM_SPECIAL_TOKENS["capitalized_token"],
                CUSTOM_SPECIAL_TOKENS["all_caps_token"]
            ],
            add_prefix_space=resolved_add_prefix_space,
            **kwargs,
        )

        self.hf_special_token_strings = {
            self.unk_token, self.cls_token, self.sep_token, self.pad_token, self.mask_token
        }
        self.custom_marker_strings = {
            self.num_token, self.cap_token, self.allcaps_token
        }

        self._last_original_inputs_for_decode: List[str] = []
        self._last_original_metadata_for_decode: List[Tuple[List[Dict[str, Any]], List[int]]] = []
        self._last_encodings_objects: List[Optional[TokenizersEncoding]] = []
        self._last_numbers_info: List[List[Dict[str, Any]]] = []

        # Improved regex pattern to correctly handle spaces and tokens
        # Zmieniono: Dodano [-+]? na początku wyrażenia regularnego dla liczb
        self.sub_word_splitter = re.compile(
            r"([-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?)" # Numbers
            r"|(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)" # Emails
            r"|(https?://\S+|www\.\S+)" # URLs
            r"|([A-Za-z_]+(?:['\-][A-Za-z_]+)*)" # Words (including contractions)
            r"|(\s+)" # Spaces (one or more)
            r"|([^\s\w\d])" # Single non-space, non-word, non-digit characters (punctuation etc.)
        )

    def train_tokenizer(self, texts_iterator: Iterator[str], vocab_size=50000, min_freq=2, show_progress=True):
        current_special_tokens = list(CUSTOM_SPECIAL_TOKENS.values())

        def pre_tokenized_texts_for_training():
            for text in texts_iterator:
                # _prepare_text_for_bpe_and_collect_metadata now inserts special tokens
                processed_tokens, _, _ = self._prepare_text_for_bpe_and_collect_metadata(text)
                yield processed_tokens

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            show_progress=show_progress,
            special_tokens=current_special_tokens
        )
        self.tokenizer.train_from_iterator(pre_tokenized_texts_for_training(), trainer=trainer)

        # Ensure all special tokens have been added to the Hugging Face wrapper vocabulary
        self.add_special_tokens({
            k: v for k, v in CUSTOM_SPECIAL_TOKENS.items() if v not in self.get_vocab()
        })

    def _prepare_text_for_bpe_and_collect_metadata(self, text: str) -> Tuple[List[str], List[Dict[str, Any]], List[int]]:
        processed_tokens_for_bpe = []
        metadata_for_original_words = []
        map_processed_idx_to_original_meta_idx = []

        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        # Ta reguła jest bardziej rygorystyczna i używana do WERYFIKACJI po wstępnym podziale.
        # Zapewnia, że przechwytuje liczby ujemne i dziesiętne/naukowe.
        number_pattern_strict = re.compile(r"[-+]?(?:0x[0-9a-fA-F]+|\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?(?:(?<=\d)[,]\d+)*)")

        for match_idx, match in enumerate(re.finditer(self.sub_word_splitter, text)):
            token_part_str = match.group(0)

            # Create a metadata entry for the original segment. This entry will refer
            # to all tokens (marker + content) derived from this original segment.
            meta_entry = {'original_value': token_part_str, 'type': 'NONE', 'start': match.start(), 'end': match.end()}

            # Add the metadata entry to the list first, so its index can be used immediately.
            metadata_for_original_words.append(meta_entry)
            current_meta_list_idx = len(metadata_for_original_words) - 1

            if token_part_str.isspace():
                meta_entry['type'] = 'SPACE'
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif url_pattern.fullmatch(token_part_str) or email_pattern.fullmatch(token_part_str):
                meta_entry['type'] = 'URL_EMAIL'
                # For URLs/Emails, we split them into characters so BPE can learn sub-parts.
                # No special marker token for them in input_ids, only content.
                for char in token_part_str:
                    processed_tokens_for_bpe.append(char)
                    map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif number_pattern_strict.fullmatch(token_part_str): # Używamy strict pattern do weryfikacji
                meta_entry['type'] = 'NUM'
                try:
                    parsed_value = None
                    numeric_format_str = 'unknown' # Default to unknown

                    if token_part_str.lower().startswith('0x'): # Hexadecimal numbers
                        parsed_value = int(token_part_str, 16)
                        meta_entry['numeric_type'] = 'int'
                        numeric_format_str = 'hexadecimal'
                    else:
                        # Normalize comma to dot for float conversion
                        normalized_num_str = token_part_str.replace(',', '.')

                        if 'e' in normalized_num_str.lower() or '.' in normalized_num_str:
                            parsed_value = float(normalized_num_str)
                            meta_entry['numeric_type'] = 'float'
                            numeric_format_str = 'scientific_notation' if 'e' in normalized_num_str.lower() else 'decimal_float'
                        else:
                            parsed_value = int(normalized_num_str)
                            meta_entry['numeric_type'] = 'int'
                            numeric_format_str = 'integer'
                    
                    meta_entry['numeric_value'] = parsed_value
                    meta_entry['numeric_format'] = numeric_format_str
                    # Store the integer ID for the format
                    meta_entry['numeric_format_id'] = self._numeric_format_to_id.get(numeric_format_str, self.numeric_format_padding_value)

                except ValueError:
                    meta_entry['numeric_value'] = None
                    meta_entry['numeric_type'] = 'unknown'
                    meta_entry['numeric_format'] = 'unknown'
                    meta_entry['numeric_format_id'] = self.numeric_format_padding_value

                # Add [NUM] token as the first in the processed stream
                processed_tokens_for_bpe.append(self.num_token)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Marker points to number metadata

                # KEY CHANGE: Add the entire number string, allowing BPE to sub-tokenize it
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Number string also points to its metadata

            elif token_part_str[0].isupper() and not token_part_str.isupper() and token_part_str.isalpha():
                meta_entry['type'] = 'CAP'
                # Add [CAP] token as the first
                processed_tokens_for_bpe.append(self.cap_token)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Marker points to word metadata
                # Then add the word itself
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Word also points to its metadata

            elif token_part_str.isupper() and len(token_part_str) > 1 and token_part_str.isalpha():
                meta_entry['type'] = 'ALLCAPS'
                # Add [ALLCAPS] token as the first
                processed_tokens_for_bpe.append(self.allcaps_token)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Marker points to word metadata
                # Then add the word itself
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Word also points to its metadata
            else: # Type 'NONE' or others (punctuation, regular words)
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Points to its metadata

        # DEBUG PRINTS
        print(f"DEBUG (prepare_text): Przetworzone tokeny dla BPE: {processed_tokens_for_bpe}")
        print(f"DEBUG (prepare_text): Metadane: {metadata_for_original_words}")
        print(f"DEBUG (prepare_text): Mapa: {map_processed_idx_to_original_meta_idx}")

        return processed_tokens_for_bpe, metadata_for_original_words, map_processed_idx_to_original_meta_idx

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, Any]] = None,
        **kwargs,
    ) -> BatchEncoding:

        # Clear data from previous run to avoid stale information
        self._last_original_inputs_for_decode = []
        self._last_original_metadata_for_decode = []
        self._last_encodings_objects = []
        self._last_numbers_info = []

        is_batched = isinstance(text, list)

        text_list = text if is_batched else [text]
        text_pair_list = text_pair if is_batched and text_pair is not None else ([text_pair] if text_pair is not None else None)

        if text_pair_list is not None and len(text_pair_list) != len(text_list):
            raise ValueError("text and text_pair must have the same number of elements in batch mode.")

        processed_texts_for_bpe = []
        processed_text_pairs_for_bpe = None

        # Store original texts and prepare for BPE, collecting metadata for each sample in the batch
        for t_idx, t_item in enumerate(text_list):
            self._last_original_inputs_for_decode.append(t_item)

            if t_item is None:
                processed_texts_for_bpe.append([])
                self._last_original_metadata_for_decode.append(([], []))
                self._last_numbers_info.append([])
                continue

            words_for_bpe, metadata_list, processed_to_original_map = \
                self._prepare_text_for_bpe_and_collect_metadata(t_item)

            processed_texts_for_bpe.append(words_for_bpe)
            self._last_original_metadata_for_decode.append((metadata_list, processed_to_original_map))

            current_numbers_info_for_seq = []
            for meta_entry in metadata_list:
                if meta_entry['type'] == 'NUM' and meta_entry.get('numeric_value') is not None:
                    current_numbers_info_for_seq.append({
                        'value': meta_entry['numeric_value'],
                        'type': meta_entry['numeric_type'],
                        'format': meta_entry['numeric_format'],
                        'format_id': meta_entry['numeric_format_id'], # Store format ID
                        'original_string': meta_entry['original_value'],
                        'original_char_span': (meta_entry['start'], meta_entry['end']),
                        'token_ids_span': None, # Will be filled after super().__call__
                        'token_ids': None,      # Will be filled after super().__call__
                    })
            self._last_numbers_info.append(current_numbers_info_for_seq)

        if text_pair_list is not None:
            processed_text_pairs_for_bpe = []
            for tp_item in text_pair_list:
                if tp_item is None:
                    processed_text_pairs_for_bpe.append([])
                    continue
                words_for_bpe_p, _, _ = self._prepare_text_for_bpe_and_collect_metadata(tp_item)
                processed_text_pairs_for_bpe.append(words_for_bpe_p)

        # Prepare input for the __call__ method of the parent PreTrainedTokenizerFast
        text_input_for_super = processed_texts_for_bpe if is_batched else processed_texts_for_bpe[0]
        text_pair_input_for_super = processed_text_pairs_for_bpe if is_batched and processed_text_pairs_for_bpe else (processed_text_pairs_for_bpe[0] if processed_text_pairs_for_bpe else None)

        # Call the __call__ method of the parent PreTrainedTokenizerFast.
        # `is_split_into_words=True` is crucial because we have already pre-tokenized the text.
        encoded_inputs = super().__call__(
            text=text_input_for_super,
            text_pair=text_pair_input_for_super,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=True,
            return_tensors=return_tensors,
            return_attention_mask=kwargs.get('return_attention_mask', True),
            return_token_type_ids=kwargs.get('return_token_type_ids', True if text_pair_input_for_super else False),
        )

        # Save the actual tokenizers.Encoding objects returned by the tokenizer.
        # These objects are necessary for mapping character/word ranges to token ID ranges.
        if hasattr(encoded_inputs, '_encodings') and encoded_inputs._encodings:
            self._last_encodings_objects = encoded_inputs._encodings
        elif isinstance(encoded_inputs, TokenizersEncoding):
            self._last_encodings_objects = [encoded_inputs]
        elif not return_tensors and not is_batched :
            # Alternative for single output without tensor: re-encode to get Encoding object
            temp_tokenizer_output = self.tokenizer.encode(
                text_input_for_super,
                pair=text_pair_input_for_super,
                add_special_tokens=add_special_tokens,
                is_pretokenized=True
            )
            self._last_encodings_objects = [temp_tokenizer_output]

        # DEBUG PRINTS
        if self._last_encodings_objects:
            print(f"DEBUG (__call__): ID kodowania: {self._last_encodings_objects[0].ids}")
            print(f"DEBUG (__call__): Tokeny kodowania: {self._last_encodings_objects[0].tokens}")
            print(f"DEBUG (__call__): ID słów kodowania: {self._last_encodings_objects[0].word_ids}")


        # --- Generate 'numeric_values' and 'numeric_formats' tensors ---
        batch_numeric_values = []
        batch_numeric_formats = [] # New tensor for numeric formats
        num_token_id = self.vocab.get(self.num_token) # Get ID for [NUM] token

        for i, encoding_obj in enumerate(self._last_encodings_objects):
            # Initialize numeric_values for this sample with padding value
            # Length corresponds to the encoded sequence after padding/truncation
            numeric_values_for_sample = torch.full(
                (len(encoding_obj.ids),),
                self.numeric_padding_value,
                dtype=torch.float32
            )
            # Initialize numeric_formats for this sample with padding value
            numeric_formats_for_sample = torch.full(
                (len(encoding_obj.ids),),
                float(self.numeric_format_padding_value), # Ensure float type for consistency with torch.full
                dtype=torch.float32 # Use float32 to match numeric_values for potential future operations
            )

            if i < len(self._last_numbers_info): # Ensure we have corresponding metadata
                current_numbers_info = self._last_numbers_info[i]
                for num_entry in current_numbers_info:
                    original_start_char, original_end_char = num_entry['original_char_span']
                    parsed_value = num_entry['value']
                    parsed_format_id = num_entry['format_id'] # Get the integer format ID

                    original_meta_idx_for_num = -1
                    for idx, meta_item in enumerate(self._last_original_metadata_for_decode[i][0]):
                        if meta_item['type'] == 'NUM' and meta_item['original_value'] == num_entry['original_string']:
                            original_meta_idx_for_num = idx
                            break

                    if original_meta_idx_for_num != -1:
                        token_indices_for_this_num_segment = []
                        for token_idx, word_id_in_encoding in enumerate(encoding_obj.word_ids):
                            if word_id_in_encoding is not None and \
                               token_idx < len(self._last_original_metadata_for_decode[i][1]) and \
                               self._last_original_metadata_for_decode[i][1][word_id_in_encoding] == original_meta_idx_for_num:

                                token_indices_for_this_num_segment.append(token_idx)
                                # Check if the current token is [NUM]
                                if encoding_obj.tokens[token_idx] == self.num_token:
                                    # Found the [NUM] token
                                    marker_token_idx = token_idx
                                    if parsed_value is not None:
                                        numeric_values_for_sample[marker_token_idx] = parsed_value
                                        numeric_formats_for_sample[marker_token_idx] = parsed_format_id # Set format ID
                                        # Determine the full span of tokens that represent the number
                                        # ([NUM] token + tokens for the number itself)
                                        start_token_span = min(token_indices_for_this_num_segment) if token_indices_for_this_num_segment else marker_token_idx
                                        end_token_span = max(token_indices_for_this_num_segment) + 1 if token_indices_for_this_num_segment else marker_token_idx + 1

                                        num_entry['token_ids_span'] = (start_token_span, end_token_span)
                                        num_entry['token_ids'] = encoding_obj.ids[start_token_span : end_token_span]
                                    break # Processed this number, move to the next
            batch_numeric_values.append(numeric_values_for_sample)
            batch_numeric_formats.append(numeric_formats_for_sample)


        # Convert list of tensors to a single tensor, matching the requested return_tensors type
        if return_tensors == "pt":
            encoded_inputs['numeric_values'] = torch.stack(batch_numeric_values, dim=0)
            encoded_inputs['numeric_formats'] = torch.stack(batch_numeric_formats, dim=0)
        elif return_tensors == "tf": # Example for TensorFlow
            import tensorflow as tf
            encoded_inputs['numeric_values'] = tf.stack(batch_numeric_values, axis=0)
            encoded_inputs['numeric_formats'] = tf.stack(batch_numeric_formats, axis=0)
        elif return_tensors == "np": # Example for NumPy
            import numpy as np
            encoded_inputs['numeric_values'] = np.stack([x.numpy() for x in batch_numeric_values], axis=0)
            encoded_inputs['numeric_formats'] = np.stack([x.numpy() for x in batch_numeric_formats], axis=0)
        else: # If no tensor type requested, return as list of lists
            encoded_inputs['numeric_values'] = [x.tolist() for x in batch_numeric_values]
            encoded_inputs['numeric_formats'] = [x.tolist() for x in batch_numeric_formats]

        return encoded_inputs

    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> str:
        # Ensure token_ids is a list of lists if batched, or a single list
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # If the input is a batch (list of lists of token IDs)
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            decoded_texts = []
            for ids_list in token_ids:
                # Call the tokenizer's decode method directly
                decoded_text = self.tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens)
                if clean_up_tokenization_spaces:
                    decoded_text = self._post_process_decoded_text(decoded_text)
                decoded_texts.append(decoded_text)
            return decoded_texts
        else:
            # If the input is a single sequence of token IDs
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            if clean_up_tokenization_spaces:
                decoded_text = self._post_process_decoded_text(decoded_text)
            return decoded_text

    def get_numeric_info(self, batch_index: int = 0) -> List[Dict[str, Any]]:
        """
        Returns detailed information about detected numbers for a specific batch item,
        including their parsed value, type, format, original string, character span,
        and resolved token ID span in the encoded sequence.
        """
        if not self._last_numbers_info or batch_index >= len(self._last_numbers_info):
            return []
        return self._last_numbers_info[batch_index]

    def get_detected_numbers_summary(self, batch_index: int = 0) -> List[str]:
        """
        Provides a concise summary of unique numbers and their formats detected
        in a specific batch item.
        """
        all_detected_numbers_summary = []
        seen_numbers_for_summary = set()

        numbers_info_for_current_batch_item = self.get_numeric_info(batch_index=batch_index)

        for num_entry in numbers_info_for_current_batch_item:
            original_pretoken_unit = num_entry.get('original_string', 'N/A')
            metadata_type = num_entry.get('format', 'NUM').upper() # Use 'format' for more detailed type

            num_key = (original_pretoken_unit, metadata_type)
            if num_key not in seen_numbers_for_summary:
                all_detected_numbers_summary.append(f"['{original_pretoken_unit}', {metadata_type}]")
                seen_numbers_for_summary.add(num_key)

        return all_detected_numbers_summary

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab(with_added_tokens=True)

    def _save_pretrained(self, save_directory: str, filename_prefix: Optional[str] = None, **kwargs):
        if filename_prefix is None:
            filename_prefix = ""

        tokenizer_file_path = os.path.join(save_directory, filename_prefix + self.vocab_files_names["tokenizer_file"])
        self.tokenizer.save(tokenizer_file_path)

        vocab_file_path = os.path.join(save_directory, filename_prefix + self.vocab_files_names["vocab_file"])
        with open(vocab_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_vocab(), f, ensure_ascii=False, indent=2)

        return (tokenizer_file_path, vocab_file_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        # Ensure numeric_padding_value is correctly set when loading
        # You might want to store this in the tokenizer's config if it's not always 0.0
        instance = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        # If numeric_padding_value needs to be loaded from config, add it here
        return instance

    def _post_process_decoded_text(self, text: str) -> str:
        """Applies common cleaning rules to decoded text."""
        # Handle common contractions
        text = text.replace(" n't", "n't")
        text = text.replace(" 're", "'re")
        text = text.replace(" 've", "'ve")
        text = text.replace(" 'll", "'ll")
        text = text.replace(" 's", "'s")
        text = text.replace(" 'm", "'m")
        text = text.replace(" 'd", "'d")

        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Remove spaces between currency symbols and numbers
        text = re.sub(r'([$€¥£])\s*([0-9])', r'\1\2', text)

        # Normalize quotes
        text = text.replace("''", '"').replace("``", '"')

        # Normalize multiple spaces to single spaces and remove leading/trailing spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text
import os
import collections
import re
import unicodedata
import json
import torch
from transformers import PreTrainedTokenizerFast, BatchEncoding
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, Regex # Ensure Regex is imported
from tokenizers.processors import TemplateProcessing
from tokenizers import Encoding as TokenizersEncoding # To avoid clash with typing.Encoding
from typing import List, Dict, Optional, Tuple, Union, Any, Iterator

# Define special tokens
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
    model_input_names = ["input_ids", "attention_mask"]

    num_token = NUMBER_TOKEN
    cap_token = CAPITALIZED_TOKEN
    allcaps_token = ALL_CAPS_TOKEN

    def __init__(self, vocab_file=None, tokenizer_file=None, **kwargs):
        self.tokenizer = None

        if tokenizer_file is not None and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.tokenizer = Tokenizer(models.BPE())
            
            special_token_values = list(CUSTOM_SPECIAL_TOKENS.values())
            self.tokenizer.add_special_tokens(special_token_values)
            
            # The pre_tokenizer handles initial splitting before our custom logic
            # This WhitespaceSplit is good for getting initial "words"
            # It should ideally be consistent with the sub_word_splitter regex.
            # Using Regex for the pre-tokenizer is usually more powerful.
            # Let's revert to using the Regex from the previous fix for consistency with sub_word_splitter.
            tokenizers_regex_pattern = Regex(
                r"(\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|https?://\S+|www\.\S+|[A-Za-z_]+(?:['\-][A-Za-z_]+)*|[^\s\w\d]|\s+)"
            )
            self.tokenizer.pre_tokenizer = pre_tokenizers.Split(
                pattern=tokenizers_regex_pattern,
                behavior="isolated" # "isolated" keeps the matched pattern as a separate token
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
        
        # Store for decode method to use original text info
        # Renamed for clarity and to avoid clash if you eventually wanted a public property.
        self._last_original_inputs_for_decode: List[str] = [] 
        # Metadata from _prepare_text_for_bpe_and_collect_metadata
        self._last_original_metadata_for_decode: List[Tuple[List[Dict[str, Any]], List[int]]] = []
        # Store the actual tokenizers.Encoding objects for detailed offset info
        self._last_encodings_objects: List[Optional[TokenizersEncoding]] = []

        # Improved regex pattern to properly handle spaces and tokens
        self.sub_word_splitter = re.compile(
            r"(\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?)" # Numbers
            r"|(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)" # Emails
            r"|(https?://\S+|www\.\S+)" # URLs
            r"|([A-Za-z_]+(?:['\-][A-Za-z_]+)*)" # Words (including contractions)
            r"|(\s+)" # Whitespace (one or more)
            r"|([^\s\w\d])" # Single non-whitespace, non-word, non-digit characters (punctuation, etc.)
        )

    def train_tokenizer(self, texts_iterator: Iterator[str], vocab_size=50000, min_freq=2, show_progress=True):
        current_special_tokens = list(CUSTOM_SPECIAL_TOKENS.values())
        
        def pre_tokenized_texts_for_training():
            for text in texts_iterator:
                # When training, we need to provide the raw words/characters that the BPE model will learn from.
                # Your `_prepare_text_for_bpe_and_collect_metadata` already does this by breaking down
                # special types (numbers, caps, urls/emails) into characters.
                processed_tokens, _, _ = self._prepare_text_for_bpe_and_collect_metadata(text)
                yield processed_tokens

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            show_progress=show_progress,
            special_tokens=current_special_tokens
        )
        # Using train_from_iterator with the generator
        self.tokenizer.train_from_iterator(pre_tokenized_texts_for_training(), trainer=trainer)
        
        # Add any special tokens that weren't picked up during training (though `special_tokens` in trainer should handle this)
        self.add_special_tokens({
            k: v for k, v in CUSTOM_SPECIAL_TOKENS.items() if v not in self.get_vocab()
        })

    def _prepare_text_for_bpe_and_collect_metadata(self, text: str) -> Tuple[List[str], List[Dict[str, Any]], List[int]]:
        """
        Pre-processes the text for BPE training/encoding and collects metadata
        about original word types and their mapping to the processed tokens.
        """
        processed_tokens_for_bpe = []
        metadata_for_original_words = []
        map_processed_idx_to_original_meta_idx = [] # Map index in `processed_tokens_for_bpe` to `metadata_for_original_words`

        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        number_pattern = re.compile(r"^(?:[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?)$")

        for match_idx, match in enumerate(re.finditer(self.sub_word_splitter, text)):
            token_part_str = match.group(0)

            # Create a metadata entry for the *original matched segment*
            meta_entry = {'original_value': token_part_str, 'type': 'NONE'}
            metadata_for_original_words.append(meta_entry)
            current_meta_list_idx = len(metadata_for_original_words) - 1

            if token_part_str.isspace():
                meta_entry['type'] = 'SPACE'
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif url_pattern.fullmatch(token_part_str):
                meta_entry['type'] = 'URL_EMAIL'
                for char in token_part_str:
                    processed_tokens_for_bpe.append(char)
                    map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif email_pattern.fullmatch(token_part_str): # Separate email check for clarity
                meta_entry['type'] = 'URL_EMAIL'
                for char in token_part_str:
                    processed_tokens_for_bpe.append(char)
                    map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif number_pattern.fullmatch(token_part_str):
                meta_entry['type'] = 'NUM'
                for char in token_part_str:
                    processed_tokens_for_bpe.append(char)
                    map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif token_part_str.isupper() and len(token_part_str) > 1 and token_part_str.isalpha():
                meta_entry['type'] = 'ALLCAPS'
                for char in token_part_str:
                    processed_tokens_for_bpe.append(char)
                    map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif token_part_str[0].isupper() and not token_part_str.isupper() and token_part_str.isalpha():
                meta_entry['type'] = 'CAP'
                for char in token_part_str:
                    processed_tokens_for_bpe.append(char)
                    map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            else:
                # Regular words and single symbols (that are not whitespace)
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
        
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
        
        # Clear previous run's stored data
        self._last_original_inputs_for_decode = []
        self._last_original_metadata_for_decode = [] 
        self._last_encodings_objects = [] 

        is_batched = isinstance(text, list)
        
        # Normalize inputs to lists for consistent processing
        text_list = text if is_batched else [text]
        text_pair_list = text_pair if is_batched and text_pair is not None else ([text_pair] if text_pair is not None else None)

        if text_pair_list is not None and len(text_pair_list) != len(text_list):
            raise ValueError("text and text_pair must have the same number of elements in batch mode.")

        processed_texts_for_bpe = []
        processed_text_pairs_for_bpe = None

        # Process primary texts
        for t_item in text_list:
            # Store original text for potential decode operations
            self._last_original_inputs_for_decode.append(t_item)

            if t_item is None: 
                processed_texts_for_bpe.append([]) 
                self._last_original_metadata_for_decode.append(([], [])) 
                continue
            
            words_for_bpe, metadata_list, processed_to_original_map = \
                self._prepare_text_for_bpe_and_collect_metadata(t_item)
            processed_texts_for_bpe.append(words_for_bpe)
            self._last_original_metadata_for_decode.append((metadata_list, processed_to_original_map))

        # Process text_pair if provided
        if text_pair_list is not None:
            processed_text_pairs_for_bpe = []
            for tp_item in text_pair_list:
                # We don't store text_pair metadata in _last_original_metadata_for_decode
                # because the current decode logic only uses data for the first sequence.
                # If needed for pair decoding, this structure would need to be enhanced.
                if tp_item is None: 
                    processed_text_pairs_for_bpe.append([])
                    continue
                words_for_bpe_p, _, _ = self._prepare_text_for_bpe_and_collect_metadata(tp_item) 
                processed_text_pairs_for_bpe.append(words_for_bpe_p)
        
        # Prepare inputs for the super class's __call__
        # Important: set is_split_into_words=True because we've already done the initial splitting
        text_input_for_super = processed_texts_for_bpe if is_batched else processed_texts_for_bpe[0]
        text_pair_input_for_super = processed_text_pairs_for_bpe if is_batched and processed_text_pairs_for_bpe else (processed_text_pairs_for_bpe[0] if processed_text_pairs_for_bpe else None)
            
        encoded_inputs = super().__call__(
            text=text_input_for_super,
            text_pair=text_pair_input_for_super,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=True, # Critical: informs Hugging Face that `text` is already split
            return_tensors=return_tensors,
            return_attention_mask=kwargs.get('return_attention_mask', True), 
            return_token_type_ids=kwargs.get('return_token_type_ids', True if text_pair_input_for_super else False),
        )
        
        # Store the actual tokenizers.Encoding objects returned by the tokenizer for decode
        if hasattr(encoded_inputs, '_encodings') and encoded_inputs._encodings: 
            self._last_encodings_objects = encoded_inputs._encodings
        elif isinstance(encoded_inputs, TokenizersEncoding): 
            self._last_encodings_objects = [encoded_inputs]
        elif not return_tensors and not is_batched : # Fallback for single, non-tensor output
            # Re-encode to get the Encoding object if not returned by super()__call__
            temp_tokenizer_output = self.tokenizer.encode(
                text_input_for_super, 
                pair=text_pair_input_for_super, 
                add_special_tokens=add_special_tokens,
                is_pretokenized=True # Inform tokenizers that input is already split
            )
            self._last_encodings_objects = [temp_tokenizer_output]
            
        return encoded_inputs

    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            decoded_texts = []
            for i, ids_list in enumerate(token_ids):
                # Retrieve the original text and its metadata for the current sequence
                original_text_for_seq = self._last_original_inputs_for_decode[i] if i < len(self._last_original_inputs_for_decode) else None
                metadata_tuple_for_seq = self._last_original_metadata_for_decode[i] if i < len(self._last_original_metadata_for_decode) else ([], [])
                encoding_obj_for_seq = self._last_encodings_objects[i] if i < len(self._last_encodings_objects) else None
                
                decoded_text = self._decode_single_sequence(
                    ids_list, 
                    original_text_for_seq, # Pass original text
                    metadata_tuple_for_seq, 
                    encoding_obj_for_seq, 
                    skip_special_tokens, 
                    clean_up_tokenization_spaces
                )
                decoded_texts.append(decoded_text)
            return decoded_texts
        else:
            # For a single sequence
            original_text_for_seq = self._last_original_inputs_for_decode[0] if self._last_original_inputs_for_decode else None
            metadata_tuple_for_seq = self._last_original_metadata_for_decode[0] if self._last_original_metadata_for_decode else ([], [])
            encoding_obj_for_seq = self._last_encodings_objects[0] if self._last_encodings_objects else None
            
            return self._decode_single_sequence(
                token_ids, 
                original_text_for_seq, # Pass original text
                metadata_tuple_for_seq, 
                encoding_obj_for_seq, 
                skip_special_tokens, 
                clean_up_tokenization_spaces
            )

    def _decode_single_sequence(
        self, 
        token_ids: List[int], 
        original_text: Optional[str], # Now explicitly passed
        metadata_tuple: Tuple[List[Dict[str, Any]], List[int]], 
        encoding_obj: Optional[TokenizersEncoding], 
        skip_special_tokens: bool, 
        clean_up_tokenization_spaces: bool 
    ) -> str:
        
        original_word_metadata_list, _ = metadata_tuple # We don't need map_processed_idx_to_original_meta_idx here with new strategy

        # If we don't have original text or metadata, fall back to basic decode
        if original_text is None or not original_word_metadata_list:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        # 1. First, perform a standard decode to get the raw text representation
        # The internal `tokenizer.decode` handles BPE merging and basic space management.
        raw_decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True) # Always decode raw, then apply markers

        # 2. Re-process the raw decoded text using your _prepare_text_for_bpe_and_collect_metadata logic
        # This will give us the "words" that the tokenizer *actually produced* and their types.
        # This is more reliable than trying to map BPE tokens back to original segments.
        reprocessed_tokens, reprocessed_metadata, _ = \
            self._prepare_text_for_bpe_and_collect_metadata(raw_decoded_text)

        reconstructed_with_markers = []
        for meta_entry in reprocessed_metadata:
            segment_value = meta_entry['original_value']
            
            if meta_entry['type'] == 'SPACE':
                reconstructed_with_markers.append(segment_value)
            elif skip_special_tokens:
                reconstructed_with_markers.append(segment_value)
            else:
                if meta_entry['type'] == 'CAP':
                    reconstructed_with_markers.append(f"{self.cap_token}{segment_value}")
                elif meta_entry['type'] == 'ALLCAPS':
                    reconstructed_with_markers.append(f"{self.allcaps_token}{segment_value}")
                elif meta_entry['type'] == 'NUM':
                    reconstructed_with_markers.append(f"{self.num_token}{segment_value}")
                elif meta_entry['type'] == 'URL_EMAIL':
                    # If you want to add a marker for URL_EMAIL, you can do it here
                    # For now, treat as regular text without a dedicated marker for simplicity if not defined
                    reconstructed_with_markers.append(segment_value)
                else: # 'NONE' type or others
                    reconstructed_with_markers.append(segment_value)

        result = "".join(reconstructed_with_markers)
        
        # Re-add special tokens at the beginning and end if not skipping them
        if not skip_special_tokens:
            pass # No explicit re-addition of CLS/SEP here, as they are often handled by the decoder or the final post-processor

        if clean_up_tokenization_spaces:
            result = self._post_process_decoded_text(result)
        
        return result

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
        return super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

    def _post_process_decoded_text(self, text: str) -> str:
        # 1. Handle contractions
        text = text.replace(" n't", "n't") 
        text = text.replace(" 're", "'re") 
        text = text.replace(" 've", "'ve") 
        text = text.replace(" 'll", "'ll") 
        text = text.replace(" 's", "'s")  
        text = text.replace(" 'm", "'m")  
        text = text.replace(" 'd", "'d")  

        # 2. Handle punctuation spacing - be more conservative
        # Remove space before certain punctuation that should attach to preceding word
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # 3. Handle currency symbols - ensure no space between symbol and number
        text = re.sub(r'([$€¥£])\s+(\d)', r'\1\2', text)
        
        # 4. Normalize quotes
        text = text.replace("''", '"').replace("``", '"')
        
        # Basic whitespace cleanup (multiple spaces to single, leading/trailing spaces)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
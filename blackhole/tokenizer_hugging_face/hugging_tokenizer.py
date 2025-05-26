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
    model_input_names = ["input_ids", "attention_mask", "numeric_values", "numeric_formats"]

    num_token = NUMBER_TOKEN
    cap_token = CAPITALIZED_TOKEN
    allcaps_token = ALL_CAPS_TOKEN

    numeric_padding_value = float('nan') # Using NaN for numeric_values padding
    numeric_format_padding_value = -1 # Using -1 for numeric_formats padding

    _numeric_format_to_id = {
        'integer': 0,
        'decimal_float': 1,
        'scientific_notation': 2,
        'hexadecimal': 3,
        'unknown': -1
    }

    def __init__(self, vocab_file=None, tokenizer_file=None, **kwargs):
        self.tokenizer = None

        # Zdefiniuj listę wszystkich tokenów specjalnych, w tym tych "dodatkowych" jak $ i %
        all_special_tokens = list(CUSTOM_SPECIAL_TOKENS.values()) + ['$', '%']

        if tokenizer_file is not None and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
            # Upewnij się, że wczytany tokenizer zna wszystkie tokeny specjalne
            tokens_to_add_to_loaded_tokenizer = [t for t in all_special_tokens if self.tokenizer.token_to_id(t) is None]
            if tokens_to_add_to_loaded_tokenizer:
                self.tokenizer.add_special_tokens(tokens_to_add_to_loaded_tokenizer)
        else:
            self.tokenizer = Tokenizer(models.BPE())

            # Dodaj WSZYSTKIE tokeny specjalne do wewnętrznego tokenizera od razu
            self.tokenizer.add_special_tokens(all_special_tokens)

            # Pre-tokenizer dla biblioteki `tokenizers` (obsługuje \p{L})
            tokenizers_regex_pattern = Regex(
                r"([-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?|\b[\p{L}0-9._%+-]+@[\p{L}0-9.-]+\.[\p{L}]{2,}\b|https?://\S+|www\.\S+|[\p{L}_]+(?:['\-][\p{L}_]+)*|[^\s\p{L}\d]|\s+)"
            )
            self.tokenizer.pre_tokenizer = pre_tokenizers.Split(
                pattern=tokenizers_regex_pattern,
                behavior="isolated"
            )

            # Sprawdzenie CLS i SEP po dodaniu wszystkich tokenów specjalnych
            cls_id = self.tokenizer.token_to_id(CUSTOM_SPECIAL_TOKENS["cls_token"])
            sep_id = self.tokenizer.token_to_id(CUSTOM_SPECIAL_TOKENS["sep_token"])

            if cls_id is None or sep_id is None:
                raise ValueError(f"Special tokens CLS ({CUSTOM_SPECIAL_TOKENS['cls_token']}) or SEP ({CUSTOM_SPECIAL_TOKENS['sep_token']}) not found in tokenizer vocabulary after initialization. This is critical.")

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
        # Usuń te parametry z kwargs, ponieważ są zarządzane przez all_special_tokens
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
            additional_special_tokens=[t for t in all_special_tokens if t not in CUSTOM_SPECIAL_TOKENS.values()], # Przekazujemy tylko te, które nie są "głównymi"
            add_prefix_space=resolved_add_prefix_space,
            **kwargs,
        )

        self.hf_special_token_strings = {
            self.unk_token, self.cls_token, self.sep_token, self.pad_token, self.mask_token
        }
        self.custom_marker_strings = {
            self.num_token, self.cap_token, self.allcaps_token, '$', '%'
        }

        self._last_original_inputs_for_decode: List[str] = []
        self._last_original_metadata_for_decode: List[Tuple[List[Dict[str, Any]], List[int]]] = []
        self._last_encodings_objects: List[Optional[TokenizersEncoding]] = []
        self._last_numbers_info: List[List[Dict[str, Any]]] = []

        # Poprawione wyrażenie regularne dla `re.compile` (używa \w i flagi re.UNICODE)
        self.sub_word_splitter = re.compile(
            r"([-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?)" # Liczby
            r"|(\b[\w0-9._%+-]+@[\w0-9.-]+\.[\w]{2,}\b)" # Emaile (używa \w dla liter)
            r"|(https?://\S+|www\.\S+)" # URL-e
            r"|(\w+(?:['\-]\w+)*)" # Słowa (używa \w dla liter, w tym polskich znaków)
            r"|(\s+)" # Spacje (jedna lub więcej)
            r"|([^\s\w\d])", # Pojedyncze znaki niebędące spacjami, słowami, cyframi (np. interpunkcja)
            re.UNICODE # Kluczowa flaga dla \w, aby pasowała do liter Unicode
        )

    def train_tokenizer(self, texts_iterator: Iterator[str], vocab_size=50000, min_freq=2, show_progress=True):
        # Upewniamy się, że wszystkie tokeny specjalne, w tym $ i %, są przekazane do trainera
        all_special_tokens_for_trainer = list(CUSTOM_SPECIAL_TOKENS.values()) + ['$', '%']
        
        def pre_tokenized_texts_for_training():
            for text in texts_iterator:
                processed_tokens, _, _ = self._prepare_text_for_bpe_and_collect_metadata(text)
                yield processed_tokens

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            show_progress=show_progress,
            special_tokens=all_special_tokens_for_trainer # Przekazujemy wszystkie tokeny specjalne do trainera
        )
        self.tokenizer.train_from_iterator(pre_tokenized_texts_for_training(), trainer=trainer)

        # Po treningu, upewnij się, że wszystkie tokeny specjalne są w słowniku HF
        # (na wypadek, gdyby min_frequency była zbyt wysoka dla niektórych w trakcie treningu,
        # choć dodanie ich do `special_tokens` w trainerze powinno temu zapobiec).
        current_vocab = self.get_vocab()
        for token in all_special_tokens_for_trainer:
            if token not in current_vocab:
                self.add_special_tokens({'additional_special_tokens': [token]})

    def _prepare_text_for_bpe_and_collect_metadata(self, text: str) -> Tuple[List[str], List[Dict[str, Any]], List[int]]:
        processed_tokens_for_bpe = []
        metadata_for_original_words = []
        map_processed_idx_to_original_meta_idx = []

        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        email_pattern = re.compile(r"\b[\w0-9._%+-]+@[\w0-9.-]+\.[\w]{2,}\b", re.UNICODE)
        number_pattern_strict = re.compile(r"[-+]?(?:0x[0-9a-fA-F]+|\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?(?:(?<=\d)[,]\d+)*)")

        for match_idx, match in enumerate(re.finditer(self.sub_word_splitter, text)):
            token_part_str = match.group(0)

            meta_entry = {'original_value': token_part_str, 'type': 'NONE', 'start': match.start(), 'end': match.end()}
            metadata_for_original_words.append(meta_entry)
            current_meta_list_idx = len(metadata_for_original_words) - 1

            if token_part_str.isspace():
                meta_entry['type'] = 'SPACE'
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif url_pattern.fullmatch(token_part_str) or email_pattern.fullmatch(token_part_str):
                meta_entry['type'] = 'URL_EMAIL'
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif number_pattern_strict.fullmatch(token_part_str):
                meta_entry['type'] = 'NUM'
                try:
                    parsed_value = None
                    numeric_format_str = 'unknown'

                    if token_part_str.lower().startswith('0x'):
                        parsed_value = int(token_part_str, 16)
                        meta_entry['numeric_type'] = 'int'
                        numeric_format_str = 'hexadecimal'
                    else:
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
                    meta_entry['numeric_format_id'] = self._numeric_format_to_id.get(numeric_format_str, self.numeric_format_padding_value)

                except ValueError:
                    meta_entry['numeric_value'] = None
                    meta_entry['numeric_type'] = 'unknown'
                    meta_entry['numeric_format'] = 'unknown'
                    meta_entry['numeric_format_id'] = self.numeric_format_padding_value

                processed_tokens_for_bpe.append(self.num_token)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)

                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)

            elif token_part_str[0].isupper() and not token_part_str.isupper() and token_part_str.isalpha():
                meta_entry['type'] = 'CAP'
                processed_tokens_for_bpe.append(self.cap_token)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)

            elif token_part_str.isupper() and len(token_part_str) > 1 and token_part_str.isalpha():
                meta_entry['type'] = 'ALLCAPS'
                processed_tokens_for_bpe.append(self.allcaps_token)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            else:
                # To jest miejsce, gdzie trafiają $ i %. Dodajemy je bezpośrednio.
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)

        # print(f"DEBUG (prepare_text): Przetworzone tokeny dla BPE: {processed_tokens_for_bpe}")
        # print(f"DEBUG (prepare_text): Metadane: {metadata_for_original_words}")
        # print(f"DEBUG (prepare_text): Mapa: {map_processed_idx_to_original_meta_idx}")

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
                        'format_id': meta_entry['numeric_format_id'],
                        'original_string': meta_entry['original_value'],
                        'original_char_span': (meta_entry['start'], meta_entry['end']),
                        'token_ids_span': None,
                        'token_ids': None,
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
            is_split_into_words=True,
            return_tensors=return_tensors,
            return_attention_mask=kwargs.get('return_attention_mask', True),
            return_token_type_ids=kwargs.get('return_token_type_ids', True if text_pair_input_for_super else False),
        )

        if hasattr(encoded_inputs, '_encodings') and encoded_inputs._encodings:
            self._last_encodings_objects = encoded_inputs._encodings
        elif isinstance(encoded_inputs, TokenizersEncoding):
            self._last_encodings_objects = [encoded_inputs]
        elif not return_tensors and not is_batched :
            temp_tokenizer_output = self.tokenizer.encode(
                text_input_for_super,
                pair=text_pair_input_for_super,
                add_special_tokens=add_special_tokens,
                is_pretokenized=True
            )
            self._last_encodings_objects = [temp_tokenizer_output]

        # print(f"DEBUG (__call__): ID kodowania: {self._last_encodings_objects[0].ids}")
        # print(f"DEBUG (__call__): Tokeny kodowania: {self._last_encodings_objects[0].tokens}")
        # print(f"DEBUG (__call__): ID słów kodowania: {self._last_encodings_objects[0].word_ids}")

        # --- Generowanie tensorów 'numeric_values' i 'numeric_formats' (POPRAWIONE) ---
        batch_numeric_values = []
        batch_numeric_formats = []
        num_token_id = self.vocab.get(self.num_token)

        for i, encoding_obj in enumerate(self._last_encodings_objects):
            numeric_values_for_sample = torch.full(
                (len(encoding_obj.ids),),
                self.numeric_padding_value,
                dtype=torch.float32 # Keep as float32, conversion to float64 for feature extraction happens in embeddings layer
            )
            numeric_formats_for_sample = torch.full(
                (len(encoding_obj.ids),),
                float(self.numeric_format_padding_value),
                dtype=torch.float32 # Keep as float32
            )

            if i < len(self._last_original_metadata_for_decode) and i < len(self._last_numbers_info):
                metadata_list, processed_to_original_map = self._last_original_metadata_for_decode[i]
                
                # Szybkie wyszukiwanie metadanych liczb po ich oryginalnym indeksie meta
                num_meta_by_original_idx = {
                    idx: meta_item for idx, meta_item in enumerate(metadata_list)
                    if meta_item['type'] == 'NUM' and meta_item.get('numeric_value') is not None
                }
                
                for token_idx, word_id_in_encoding in enumerate(encoding_obj.word_ids):
                    if encoding_obj.ids[token_idx] == num_token_id and word_id_in_encoding is not None:
                        if word_id_in_encoding < len(processed_to_original_map):
                            original_meta_idx = processed_to_original_map[word_id_in_encoding]
                            
                            if original_meta_idx in num_meta_by_original_idx:
                                num_meta_entry = num_meta_by_original_idx[original_meta_idx]
                                
                                parsed_value = num_meta_entry['numeric_value']
                                parsed_format_id = num_meta_entry['numeric_format_id']

                                if parsed_value is not None:
                                    numeric_values_for_sample[token_idx] = parsed_value
                                    numeric_formats_for_sample[token_idx] = parsed_format_id
                                    
            batch_numeric_values.append(numeric_values_for_sample)
            batch_numeric_formats.append(numeric_formats_for_sample)

        if return_tensors == "pt":
            encoded_inputs['numeric_values'] = torch.stack(batch_numeric_values, dim=0)
            encoded_inputs['numeric_formats'] = torch.stack(batch_numeric_formats, dim=0)
        elif return_tensors == "tf":
            import tensorflow as tf
            encoded_inputs['numeric_values'] = tf.stack(batch_numeric_values, axis=0)
            encoded_inputs['numeric_formats'] = tf.stack(batch_numeric_formats, axis=0)
        elif return_tensors == "np":
            import numpy as np
            encoded_inputs['numeric_values'] = np.stack([x.numpy() for x in batch_numeric_values], axis=0)
            encoded_inputs['numeric_formats'] = np.stack([x.numpy() for x in batch_numeric_formats], axis=0)
        else:
            encoded_inputs['numeric_values'] = [x.tolist() for x in batch_numeric_values]
            encoded_inputs['numeric_formats'] = [x.tolist() for x in batch_numeric_formats]

        return encoded_inputs

    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True, # This controls *your* _post_process_decoded_text
        **kwargs,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            decoded_texts = []
            for ids_list in token_ids:
                # WAŻNA ZMIANA: Przekazujemy clean_up_tokenization_spaces=False do bazowego dekodera
                # Chcemy, aby to nasza metoda _post_process_decoded_text zajęła się czyszczeniem.
                decoded_text = self.tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)
                if clean_up_tokenization_spaces:
                    decoded_text = self._post_process_decoded_text(decoded_text)
                decoded_texts.append(decoded_text)
            return decoded_texts
        else:
            # WAŻNA ZMIANA: Przekazujemy clean_up_tokenization_spaces=False do bazowego dekodera
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            if clean_up_tokenization_spaces:
                decoded_text = self._post_process_decoded_text(decoded_text)
            return decoded_text

    def get_numeric_info(self, batch_index: int = 0) -> List[Dict[str, Any]]:
        if not self._last_numbers_info or batch_index >= len(self._last_numbers_info):
            return []
        return self._last_numbers_info[batch_index]

    def get_detected_numbers_summary(self, batch_index: int = 0) -> List[str]:
        all_detected_numbers_summary = []
        seen_numbers_for_summary = set()

        numbers_info_for_current_batch_item = self.get_numeric_info(batch_index=batch_index)

        for num_entry in numbers_info_for_current_batch_item:
            original_pretoken_unit = num_entry.get('original_string', 'N/A')
            metadata_type = num_entry.get('format', 'NUM').upper()

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
        instance = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return instance

    def _post_process_decoded_text(self, text: str) -> str:
        # Standardowe czyszczenie spacji i interpunkcji
        text = text.replace(" n't", "n't")
        text = text.replace(" 're", "'re")
        text = text.replace(" 've", "'ve")
        text = text.replace(" 'll", "'ll")
        text = text.replace(" 's", "'s")
        text = text.replace(" 'm", "'m")
        text = text.replace(" 'd", "'d")

        # Usuń spacje przed znakami interpunkcyjnymi
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Nowa reguła: Usuń spację przed znakiem procentu, jeśli jest liczba
        # To jest kluczowe dla "1.5 %" -> "1.5%"
        text = re.sub(r'(\d(?:\.\d+)?)\s+(%)', r'\1\2', text)
        
        text = text.replace("''", '"').replace("``", '"')
        
        # Normalizuj wszystkie sekwencje spacji do pojedynczej spacji i usuń z początku/końca
        text = re.sub(r'\s+', ' ', text).strip()

        return text
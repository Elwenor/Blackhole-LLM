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
    model_input_names = ["input_ids", "attention_mask", "numeric_values"] # Dodano numeric_values

    num_token = NUMBER_TOKEN
    cap_token = CAPITALIZED_TOKEN
    allcaps_token = ALL_CAPS_TOKEN

    # Definiowanie wartości dopełnienia dla tensora numeric_values
    numeric_padding_value = 0.0

    def __init__(self, vocab_file=None, tokenizer_file=None, **kwargs):
        self.tokenizer = None

        if tokenizer_file is not None and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.tokenizer = Tokenizer(models.BPE())

            special_token_values = list(CUSTOM_SPECIAL_TOKENS.values())
            self.tokenizer.add_special_tokens(special_token_values)

            # Pre-tokenizer dzieli surowy tekst na segmenty (słowa, liczby, interpunkcja, spacje)
            # Właściwe wstawianie specjalnych tokenów znacznikowych ([NUM], [CAP], [ALLCAPS])
            # nastąpi w _prepare_text_for_bpe_and_collect_metadata
            tokenizers_regex_pattern = Regex(
                r"(\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|https?://\S+|www\.\S+|[A-Za-z_]+(?:['\-][A-Za-z_]+)*|[^\s\w\d]|\s+)"
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

            # Post-processor dodaje tokeny CLS/SEP wokół sekwencji
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

        # Ulepszony wzorzec regex do prawidłowego obsługiwania spacji i tokenów
        self.sub_word_splitter = re.compile(
            r"(\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?)" # Liczby
            r"|(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)" # Emaile
            r"|(https?://\S+|www\.\S+)" # URL-e
            r"|([A-Za-z_]+(?:['\-][A-Za-z_]+)*)" # Słowa (w tym skróty)
            r"|(\s+)" # Spacje (jedna lub więcej)
            r"|([^\s\w\d])" # Pojedyncze znaki niebędące spacją, słowem, cyfrą (interpunkcja itp.)
        )

    def train_tokenizer(self, texts_iterator: Iterator[str], vocab_size=50000, min_freq=2, show_progress=True):
        current_special_tokens = list(CUSTOM_SPECIAL_TOKENS.values())

        def pre_tokenized_texts_for_training():
            for text in texts_iterator:
                # _prepare_text_for_bpe_and_collect_metadata teraz wstawia specjalne tokeny
                processed_tokens, _, _ = self._prepare_text_for_bpe_and_collect_metadata(text)
                yield processed_tokens

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            show_progress=show_progress,
            special_tokens=current_special_tokens
        )
        self.tokenizer.train_from_iterator(pre_tokenized_texts_for_training(), trainer=trainer)

        # Upewnij się, że wszystkie specjalne tokeny zostały dodane do słownika wrappera Hugging Face
        self.add_special_tokens({
            k: v for k, v in CUSTOM_SPECIAL_TOKENS.items() if v not in self.get_vocab()
        })

    def _prepare_text_for_bpe_and_collect_metadata(self, text: str) -> Tuple[List[str], List[Dict[str, Any]], List[int]]:
        """
        Pre-procesuje tekst dla treningu/kodowania BPE i zbiera metadane
        o oryginalnych typach słów i ich mapowaniu na przetworzone tokeny.
        Wstawia również specjalne tokeny znacznikowe ([NUM], [CAP], [ALLCAPS]) do strumienia
        przetworzonych tokenów, który zostanie przekazany do modelu BPE.
        """
        processed_tokens_for_bpe = []
        metadata_for_original_words = []
        # map_processed_idx_to_original_meta_idx mapuje indeks w `processed_tokens_for_bpe`
        # na indeks w `metadata_for_original_words`. Jest to kluczowe do łączenia tokenów BPE
        # z ich oryginalnymi jednostkami semantycznymi i powiązanymi metadanymi.
        map_processed_idx_to_original_meta_idx = []

        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        # Poprawiony wzorzec dla liczb, aby obsługiwał również szesnastkowe (0x...)
        # oraz bardziej elastycznie przecinki jako separatory tysięcy (choć później normalizujemy do kropki)
        number_pattern_strict = re.compile(r"[-+]?(?:0x[0-9a-fA-F]+|\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?(?:(?<=\d)[,]\d+)*)")

        for match_idx, match in enumerate(re.finditer(self.sub_word_splitter, text)):
            token_part_str = match.group(0)

            # Utwórz wpis metadanych dla oryginalnego segmentu. Ten wpis będzie odwoływał się
            # do wszystkich tokenów (znacznika + zawartości) pochodzących z tego oryginalnego segmentu.
            meta_entry = {'original_value': token_part_str, 'type': 'NONE', 'start': match.start(), 'end': match.end()}

            # Dodaj wpis metadanych do listy jako pierwszy, aby jego indeks mógł być od razu użyty.
            metadata_for_original_words.append(meta_entry)
            current_meta_list_idx = len(metadata_for_original_words) - 1

            if token_part_str.isspace():
                meta_entry['type'] = 'SPACE'
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif url_pattern.fullmatch(token_part_str) or email_pattern.fullmatch(token_part_str):
                meta_entry['type'] = 'URL_EMAIL'
                # Dla URL-i/Emaili, dzielimy je na znaki, aby BPE mógł nauczyć się sub-części.
                # Brak specjalnego tokenu znacznika dla nich w input_ids, tylko zawartość.
                for char in token_part_str:
                    processed_tokens_for_bpe.append(char)
                    map_processed_idx_to_original_meta_idx.append(current_meta_list_idx)
            elif number_pattern_strict.fullmatch(token_part_str):
                meta_entry['type'] = 'NUM'
                try:
                    parsed_value = None
                    if token_part_str.lower().startswith('0x'): # Liczby szesnastkowe
                        parsed_value = int(token_part_str, 16)
                        meta_entry['numeric_type'] = 'int'
                        meta_entry['numeric_format'] = 'hexadecimal'
                    else:
                        # Normalizuj przecinek na kropkę dla konwersji na float
                        normalized_num_str = token_part_str.replace(',', '.')

                        if 'e' in normalized_num_str.lower() or '.' in normalized_num_str:
                            parsed_value = float(normalized_num_str)
                            meta_entry['numeric_type'] = 'float'
                            meta_entry['numeric_format'] = 'scientific_notation' if 'e' in normalized_num_str.lower() else 'decimal_float'
                        else:
                            parsed_value = int(normalized_num_str)
                            meta_entry['numeric_type'] = 'int'
                            meta_entry['numeric_format'] = 'integer'
                    meta_entry['numeric_value'] = parsed_value
                except ValueError:
                    meta_entry['numeric_value'] = None
                    meta_entry['numeric_type'] = 'unknown'
                    meta_entry['numeric_format'] = 'unknown'

                # Dodaj token [NUM] jako pierwszy do przetworzonego strumienia
                processed_tokens_for_bpe.append(self.num_token)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Marker wskazuje na metadane liczby

                # KLUCZOWA ZMIANA: Dodaj cały ciąg znaków liczby, pozwalając BPE na sub-tokenizację
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Ciąg znaków liczby również wskazuje na jej metadane

            elif token_part_str[0].isupper() and not token_part_str.isupper() and token_part_str.isalpha():
                meta_entry['type'] = 'CAP'
                # Dodaj token [CAP] jako pierwszy
                processed_tokens_for_bpe.append(self.cap_token)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Marker wskazuje na metadane słowa
                # Następnie dodaj samo słowo
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Słowo również wskazuje na swoje metadane

            elif token_part_str.isupper() and len(token_part_str) > 1 and token_part_str.isalpha():
                meta_entry['type'] = 'ALLCAPS'
                # Dodaj token [ALLCAPS] jako pierwszy
                processed_tokens_for_bpe.append(self.allcaps_token)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Marker wskazuje na metadane słowa
                # Następnie dodaj samo słowo
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Słowo również wskazuje na swoje metadane
            else: # Typ 'NONE' lub inne (interpunkcja, zwykłe słowa)
                processed_tokens_for_bpe.append(token_part_str)
                map_processed_idx_to_original_meta_idx.append(current_meta_list_idx) # Wskazuje na swoje metadane

        # WYDRUKI DEBUGOWANIA
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

        # Wyczyść dane z poprzedniego uruchomienia, aby uniknąć przestarzałych informacji
        self._last_original_inputs_for_decode = []
        self._last_original_metadata_for_decode = []
        self._last_encodings_objects = []
        self._last_numbers_info = []

        is_batched = isinstance(text, list)

        text_list = text if is_batched else [text]
        text_pair_list = text_pair if is_batched and text_pair is not None else ([text_pair] if text_pair is not None else None)

        if text_pair_list is not None and len(text_pair_list) != len(text_list):
            raise ValueError("text i text_pair muszą mieć tę samą liczbę elementów w trybie wsadowym.")

        processed_texts_for_bpe = []
        processed_text_pairs_for_bpe = None

        # Przechowaj oryginalne teksty i przygotuj do BPE, zbierając metadane dla każdej próbki w partii
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
                        'original_string': meta_entry['original_value'],
                        'original_char_span': (meta_entry['start'], meta_entry['end']),
                        'token_ids_span': None, # Zostanie wypełnione po super().__call__
                        'token_ids': None,      # Zostanie wypełnione po super().__call__
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

        # Przygotuj dane wejściowe dla metody __call__ nadrzędnego PreTrainedTokenizerFast
        text_input_for_super = processed_texts_for_bpe if is_batched else processed_texts_for_bpe[0]
        text_pair_input_for_super = processed_text_pairs_for_bpe if is_batched and processed_text_pairs_for_bpe else (processed_text_pairs_for_bpe[0] if processed_text_pairs_for_bpe else None)

        # Wywołaj metodę __call__ nadrzędnego PreTrainedTokenizerFast.
        # `is_split_into_words=True` jest kluczowe, ponieważ już wstępnie tokenizowaliśmy tekst.
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

        # Zapisz rzeczywiste obiekty tokenizers.Encoding zwrócone przez tokenizer.
        # Obiekty te są niezbędne do mapowania zakresów znaków/słów na zakresy ID tokenów.
        if hasattr(encoded_inputs, '_encodings') and encoded_inputs._encodings:
            self._last_encodings_objects = encoded_inputs._encodings
        elif isinstance(encoded_inputs, TokenizersEncoding):
            self._last_encodings_objects = [encoded_inputs]
        elif not return_tensors and not is_batched :
            # Alternatywa dla pojedynczego wyjścia bez tensora: ponownie zakoduj, aby uzyskać obiekt Encoding
            temp_tokenizer_output = self.tokenizer.encode(
                text_input_for_super,
                pair=text_pair_input_for_super,
                add_special_tokens=add_special_tokens,
                is_pretokenized=True
            )
            self._last_encodings_objects = [temp_tokenizer_output]

        # WYDRUKI DEBUGOWANIA
        if self._last_encodings_objects:
            print(f"DEBUG (__call__): ID kodowania: {self._last_encodings_objects[0].ids}")
            print(f"DEBUG (__call__): Tokeny kodowania: {self._last_encodings_objects[0].tokens}")
            print(f"DEBUG (__call__): ID słów kodowania: {self._last_encodings_objects[0].word_ids}")


        # --- Generuj tensor 'numeric_values' ---
        batch_numeric_values = []
        num_token_id = self.vocab.get(self.num_token) # Pobierz ID dla tokenu [NUM]

        for i, encoding_obj in enumerate(self._last_encodings_objects):
            # Zainicjuj numeric_values dla tej próbki wartością dopełnienia
            # Długość odpowiada zakodowanej sekwencji po dopełnieniu/obcięciu
            numeric_values_for_sample = torch.full(
                (len(encoding_obj.ids),),
                self.numeric_padding_value,
                dtype=torch.float32
            )

            if i < len(self._last_numbers_info): # Upewnij się, że mamy odpowiadające metadane
                current_numbers_info = self._last_numbers_info[i]
                for num_entry in current_numbers_info:
                    original_start_char, original_end_char = num_entry['original_char_span']
                    parsed_value = num_entry['value']

                    # Znajdź word_id odpowiadające oryginalnemu zakresowi znaków liczby.
                    # Ten `word_id` odnosi się do wstępnie tokenizowanego segmentu (np. "[NUM]123.45").
                    # Ważne: char_to_word działa na podstawie oryginalnego (surowego) tekstu
                    # i mapuje go na słowa, które *zostały przekazane do tokenizer.encode*.
                    # To, co przekazujemy do tokenizer.encode, to lista stringów z processed_tokens_for_bpe.
                    # Działanie char_to_word może być tutaj nieintuicyjne, ponieważ
                    # tokeny specjalne ([NUM], [CAP]) nie mają bezpośredniego mapowania na znaki
                    # w oryginalnym tekście, a co za tym idzie, word_id dla nich będzie None.
                    # Musimy znaleźć token [NUM] w faktycznych tokenach kodowania, a następnie użyć jego pozycji.

                    # Zamiast polegać na char_to_word dla markera [NUM], szukamy go bezpośrednio
                    # w tokenach wynikowych i dopasowujemy do oryginalnych danych.
                    # Musimy przeglądać listę tokenów z encodings.tokens, aby znaleźć token [NUM],
                    # a następnie potwierdzić, czy następne tokeny odpowiadają oryginalnej liczbie.

                    # Ulepszone podejście: Przejdź przez word_ids z enkodowania,
                    # które mapuje tokeny BPE na indeksy słów w processed_tokens_for_bpe.
                    # Następnie użyj map_processed_idx_to_original_meta_idx,
                    # aby połączyć to z metadata_for_original_words.

                    # Jeśli istnieje token [NUM], będzie on miał jakiś word_id,
                    # a następnie kolejny token (cała liczba) będzie miał ten sam word_id.
                    # Szukamy word_id, które odpowiada naszej liczbie w `metadata_for_original_words`.

                    # Znajdź indeks w metadata_for_original_words dla obecnej liczby
                    original_meta_idx_for_num = -1
                    for idx, meta_item in enumerate(self._last_original_metadata_for_decode[i][0]):
                        if meta_item['type'] == 'NUM' and meta_item['original_value'] == num_entry['original_string']:
                            original_meta_idx_for_num = idx
                            break

                    if original_meta_idx_for_num != -1:
                        # Przeglądaj tokeny w encoding_obj, aby znaleźć te, które mapują do tego word_id
                        # i zawierają token [NUM].
                        token_indices_for_this_num_segment = []
                        for token_idx, word_id_in_encoding in enumerate(encoding_obj.word_ids):
                            # word_id_in_encoding to indeks z listy `processed_tokens_for_bpe`
                            # Musimy mapować ten indeks na `original_meta_idx`
                            if word_id_in_encoding is not None and \
                               token_idx < len(self._last_original_metadata_for_decode[i][1]) and \
                               self._last_original_metadata_for_decode[i][1][word_id_in_encoding] == original_meta_idx_for_num:

                                token_indices_for_this_num_segment.append(token_idx)
                                # Sprawdź, czy obecny token to [NUM]
                                if encoding_obj.tokens[token_idx] == self.num_token:
                                    # Znalazłeś token [NUM]
                                    marker_token_idx = token_idx
                                    if parsed_value is not None:
                                        numeric_values_for_sample[marker_token_idx] = parsed_value
                                        # Określ pełny zakres tokenów, które reprezentują liczbę
                                        # (token [NUM] + tokeny dla samej liczby)
                                        start_token_span = min(token_indices_for_this_num_segment) if token_indices_for_this_num_segment else marker_token_idx
                                        end_token_span = max(token_indices_for_this_num_segment) + 1 if token_indices_for_this_num_segment else marker_token_idx + 1

                                        num_entry['token_ids_span'] = (start_token_span, end_token_span)
                                        num_entry['token_ids'] = encoding_obj.ids[start_token_span : end_token_span]
                                    break # Przetworzono tę liczbę, przejdź do następnej
            batch_numeric_values.append(numeric_values_for_sample)


        # Konwertuj listę tensorów na pojedynczy tensor, dopasowując do żądanego typu return_tensors
        if return_tensors == "pt":
            encoded_inputs['numeric_values'] = torch.stack(batch_numeric_values, dim=0)
        elif return_tensors == "tf": # Przykład dla TensorFlow
            import tensorflow as tf
            encoded_inputs['numeric_values'] = tf.stack(batch_numeric_values, axis=0)
        elif return_tensors == "np": # Przykład dla NumPy
            import numpy as np
            encoded_inputs['numeric_values'] = np.stack([x.numpy() for x in batch_numeric_values], axis=0)
        else: # Jeśli nie żądano typu tensora, zwróć jako listę list
            encoded_inputs['numeric_values'] = [x.tolist() for x in batch_numeric_values]

        return encoded_inputs

    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> str:
        # Upewnij się, że token_ids to lista list, jeśli jest batched, lub pojedyncza lista
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Jeśli wejście jest batchem (lista list ID tokenów)
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            decoded_texts = []
            for ids_list in token_ids:
                # Wywołaj bezpośrednio metodę decode tokenizer'a
                decoded_text = self.tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens)
                if clean_up_tokenization_spaces:
                    decoded_text = self._post_process_decoded_text(decoded_text)
                decoded_texts.append(decoded_text)
            return decoded_texts
        else:
            # Jeśli wejście to pojedyncza sekwencja ID tokenów
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            if clean_up_tokenization_spaces:
                decoded_text = self._post_process_decoded_text(decoded_text)
            return decoded_text

    def get_numeric_info(self, batch_index: int = 0) -> List[Dict[str, Any]]:
        """
        Zwraca szczegółowe informacje o wykrytych liczbach dla konkretnego elementu partii,
        w tym ich sparsowaną wartość, typ, format, oryginalny ciąg, zakres znaków
        i rozwiązany zakres ID tokenów w zakodowanej sekwencji.
        """
        if not self._last_numbers_info or batch_index >= len(self._last_numbers_info):
            return []
        return self._last_numbers_info[batch_index]

    def get_detected_numbers_summary(self, batch_index: int = 0) -> List[str]:
        """
        Dostarcza zwięzłe podsumowanie unikalnych liczb i ich formatów wykrytych
        w konkretnym elemencie partii.
        """
        all_detected_numbers_summary = []
        seen_numbers_for_summary = set()

        numbers_info_for_current_batch_item = self.get_numeric_info(batch_index=batch_index)

        for num_entry in numbers_info_for_current_batch_item:
            original_pretoken_unit = num_entry.get('original_string', 'N/A')
            metadata_type = num_entry.get('format', 'NUM').upper() # Używamy 'format' dla bardziej szczegółowego typu

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
        # Upewnij się, że numeric_padding_value jest poprawnie ustawione podczas ładowania
        # Możesz chcieć przechowywać to w konfiguracji tokenizer'a, jeśli nie zawsze jest to 0.0
        instance = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        # Jeśli numeric_padding_value musi być załadowane z konfiguracji, dodaj to tutaj
        return instance

    def _post_process_decoded_text(self, text: str) -> str:
        """Stosuje wspólne zasady czyszczenia do dekodowanego tekstu."""
        # Obsłuż popularne skróty
        text = text.replace(" n't", "n't")
        text = text.replace(" 're", "'re")
        text = text.replace(" 've", "'ve")
        text = text.replace(" 'll", "'ll")
        text = text.replace(" 's", "'s")
        text = text.replace(" 'm", "'m")
        text = text.replace(" 'd", "'d")

        # Usuń spacje przed interpunkcją
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Usuń spacje między symbolami walut i liczbami
        text = re.sub(r'([$€¥£])\s*([0-9])', r'\1\2', text)

        # Znormalizuj cudzysłowy
        text = text.replace("''", '"').replace("``", '"')

        # Znormalizuj wielokrotne spacje do pojedynczych spacji i usuń spacje wiodące/końcowe
        text = re.sub(r'\s+', ' ', text).strip()

        return text
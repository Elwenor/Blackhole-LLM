# scripts/training/data_processing.py

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

class Seq2SeqTextNumericDataset(Dataset):
    """
    A PyTorch Dataset for Seq2Seq tasks that processes text and numeric features.
    It takes raw text from a Hugging Face Dataset and uses the custom
    BlackholeTokenizer to prepare model inputs and labels.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, hf_dataset: HFDataset, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        print(f"DEBUG: Loading from data_processing.py: {__file__}")
        # Get a single raw data point (question and answer).
        example = self.hf_dataset[idx]
        question = example['question']
        answer = example['answer']

        # Tokenize the source text (question) for the encoder.
        encoder_tokenized = self.tokenizer(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_numeric_features=True
        )

        # Tokenize the target text (answer) to create the labels for the decoder.
        decoder_tokenized = self.tokenizer(
            answer,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_numeric_features=True
        )

        # Create the final dictionary for the model
        processed_example = {
            "encoder_input_ids": encoder_tokenized.input_ids.squeeze(0),
            "encoder_attention_mask": encoder_tokenized.attention_mask.squeeze(0),
            "encoder_numeric_values": encoder_tokenized.numeric_values.squeeze(0),
            "encoder_numeric_formats": encoder_tokenized.numeric_formats.squeeze(0),
            "labels": decoder_tokenized.input_ids.squeeze(0),
            "decoder_numeric_values": decoder_tokenized.numeric_values.squeeze(0),
            "decoder_numeric_formats": decoder_tokenized.numeric_formats.squeeze(0)
        }

        logger.debug(f"__getitem__ for index {idx} returning keys: {list(processed_example.keys())}")
        for k, v in processed_example.items():
            if isinstance(v, torch.Tensor):
                logger.debug(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                logger.debug(f"  {k}: {v}")

        return processed_example
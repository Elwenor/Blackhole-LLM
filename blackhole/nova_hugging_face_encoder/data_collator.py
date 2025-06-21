import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling

from blackhole.tokenizer_hugging_face import CUSTOM_SPECIAL_TOKENS 

@dataclass
class BlackholeDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    seed: Optional[int] = None
    max_length: Optional[int] = None 

    def __post_init__(self):
        if not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None:
            raise ValueError(
                "Tokenizator musi mieć zdefiniowany `mask_token` dla Masked Language Modeling. "
                "Upewnij się, że `CUSTOM_SPECIAL_TOKENS['mask_token']` jest dodany i tokenizer go używa."
            )
        
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(CUSTOM_SPECIAL_TOKENS["mask_token"])
        self.num_token_id = self.tokenizer.convert_tokens_to_ids(CUSTOM_SPECIAL_TOKENS["number_token"])
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = self.tokenizer.vocab_size

        if self.seed is not None:
            self.generator = torch.Generator().manual_seed(self.seed)
        else:
            self.generator = torch.Generator()


    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        processed_numeric_values = []
        processed_numeric_formats = []

        for e in examples:
            nv = e.pop("numeric_values", [])
            nf = e.pop("numeric_formats", [])
            
            if isinstance(nv, list) and len(nv) > 0 and isinstance(nv[0], list):
                processed_numeric_values.append([item for sublist in nv for item in sublist])
            else:
                processed_numeric_values.append(nv)
            
            if isinstance(nf, list) and len(nf) > 0 and isinstance(nf[0], list):
                processed_numeric_formats.append([item for sublist in nf for item in sublist])
            else:
                processed_numeric_formats.append(nf)

        batch = super().__call__(examples, return_tensors=self.return_tensors)

        max_length = batch["input_ids"].shape[1]
        batch_size = batch["input_ids"].shape[0]
        
        padded_numeric_values_tensor = torch.full(
            (batch_size, max_length),
            float('nan'),
            dtype=torch.float32,
            device=batch["input_ids"].device
        )
        padded_numeric_formats_tensor = torch.full(
            (batch_size, max_length),
            -1,
            dtype=torch.long,
            device=batch["input_ids"].device
        )

        for i in range(batch_size):
            # POPRAWKA ZASTOSOWANA TUTAJ: Dodano .clone().detach()
            # Najpierw przekształć listę/inne dane na tensor, jeśli jeszcze nim nie są
            current_numeric_values = torch.tensor(processed_numeric_values[i], dtype=torch.float32)
            current_numeric_formats = torch.tensor(processed_numeric_formats[i], dtype=torch.long)

            current_numeric_values = current_numeric_values.clone().detach()
            current_numeric_formats = current_numeric_formats.clone().detach()

            if current_numeric_values.ndim > 1 and current_numeric_values.shape[0] == 1:
                current_numeric_values = current_numeric_values.squeeze(0) 

            if current_numeric_formats.ndim > 1 and current_numeric_formats.shape[0] == 1:
                current_numeric_formats = current_numeric_formats.squeeze(0)
            
            actual_len = current_numeric_values.shape[0] 
            copy_len = min(actual_len, max_length)
            
            if copy_len > 0:
                try:
                    padded_numeric_values_tensor[i, :copy_len] = current_numeric_values[:copy_len]
                except RuntimeError as e:
                    print(f"ERROR during numeric_values assignment for example {i}: {e}")
                    print(f"  Target slice shape: {padded_numeric_values_tensor[i, :copy_len].shape}")
                    print(f"  Source tensor shape: {current_numeric_values[:copy_len].shape}")
                    raise
                
                try:
                    padded_numeric_formats_tensor[i, :copy_len] = current_numeric_formats[:copy_len]
                except RuntimeError as e:
                    print(f"ERROR during numeric_formats assignment for example {i}: {e}")
                    print(f"  Target slice shape: {padded_numeric_formats_tensor[i, :copy_len].shape}")
                    print(f"  Source tensor shape: {current_numeric_formats[:copy_len].shape}")
                    raise

        batch["numeric_values"] = padded_numeric_values_tensor
        batch["numeric_formats"] = padded_numeric_formats_tensor
        
        return batch
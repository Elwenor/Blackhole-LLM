import torch
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List, Callable

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_scheduler
from transformers.utils import logging

# Importuj swoje komponenty z odpowiednich ścieżek
from blackhole.nova_hugging_face_encoder.modeling_nova import BlackholeForMaskedLM, BlackholeForSequenceClassification
from blackhole.nova_hugging_face_encoder.configuration_nova import BlackholeConfig

# Poprawiona ścieżka importu dla tokenizatora
from blackhole.tokenizer_hugging_face import BlackholeTokenizer
# Poprawiona ścieżka importu dla data_collator
from blackhole.nova_hugging_face_encoder import BlackholeDataCollatorForLanguageModeling

logger = logging.get_logger(__name__)

# --- Klasa TrainingArguments ---
@dataclass
class TrainingArguments:
    """
    Argumenty konfiguracyjne dla treningu modelu Blackhole.
    """
    output_dir: str = field(
        metadata={"help": "Katalog wyjściowy dla checkpointów modelu i wyników treningu."}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Rozmiar batcha dla treningu na każdym urządzeniu."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Rozmiar batcha dla ewaluacji na każdym urządzeniu."}
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Liczba epok treningowych."}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "Początkowa szybkość uczenia dla optymalizatora AdamW."}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Waga regularyzacji L2 dla optymalizatora."}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Liczba kroków rozgrzewkowych dla szybkości uczenia."}
    )
    logging_steps: int = field(
        default=100, metadata={"help": "Liczba kroków po których logowane są statystyki treningu."}
    )
    eval_steps: int = field(
        default=500, metadata={"help": "Liczba kroków po których przeprowadzana jest ewaluacja."}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Liczba kroków po których zapisywany jest checkpoint modelu."}
    )
    seed: int = field(
        default=42, metadata={"help": "Ziarno dla inicjalizacji generatorów liczb losowych."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Liczba kroków, przez którą gradienty są akumulowane przed wykonaniem kroku optymalizatora."}
    )
    fp16: bool = field(
        default=False, metadata={"help": "Czy używać treningu w mieszanej precyzji (float16)."}
    )

# --- Klasa NovaTrainer ---
class NovaTrainer:
    """
    Trener dla modelu Blackhole, zarządzający cyklem treningowym i ewaluacyjnym.
    """

    def __init__(
        self,
        model: Union[BlackholeForMaskedLM, BlackholeForSequenceClassification],
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        tokenizer: Optional[BlackholeTokenizer] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
    ):
        if model is None:
            raise ValueError("`model` musi być dostarczony do NovaTrainer.")
        if args is None:
            raise ValueError("`args` (TrainingArguments) muszą być dostarczone do NovaTrainer.")

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Logowanie rozmiaru słownika modelu i tokenizatora
        if self.tokenizer:
            logger.info(f"Tokenizator: vocab_size={len(self.tokenizer)}")
        logger.info(f"Model: vocab_size={self.model.config.vocab_size}")

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is None and train_dataset is not None:
            num_update_steps_per_epoch = math.ceil(len(train_dataset) / (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps))
            num_training_steps = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=num_training_steps,
            )
        
        self.scaler = None
        if self.args.fp16 and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Używanie NVIDIA Apex AMP (mieszana precyzja).")
        elif self.args.fp16 and self.device.type == "cpu":
            logger.warning("Mieszana precyzja (fp16) nie jest obsługiwana na CPU. Wyłączanie fp16.")
            self.args.fp16 = False


    def train(self):
        """
        Główna metoda uruchamiająca proces treningu.
        """
        if self.train_dataset is None:
            raise ValueError("`train_dataset` musi być dostarczony do metody `train`.")
        
        # Jeśli data_collator nie został przekazany, tworzymy go tutaj
        if self.data_collator is None:
            if self.tokenizer is None:
                raise ValueError("`tokenizer` musi być dostarczony do NovaTrainer, jeśli `data_collator` nie jest jawnie inicjowany.")
            
            # Upewnij się, że max_length jest zgodne z modelem i tokenizatorem
            # model_max_length = self.model.config.max_position_embeddings # To może być inne niż max_length tokenizatora
            # Lepiej użyć max_length z tokenizatora lub jasno zdefiniować.
            # Jeśli max_length jest używane do truncate'owania podczas kolacji, powinno być zgodne z modelem.
            # Przyjmuję, że config.max_position_embeddings jest odpowiednie.
            model_max_length = self.model.config.max_position_embeddings
            
            self.data_collator = BlackholeDataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=0.15,
                pad_to_multiple_of=8,
                seed=self.args.seed,
                max_length=model_max_length # Ważne, aby upewnić się, że to jest stosowane
            )
        
        logger.info("***** Rozpynanie treningu *****")
        logger.info(f" Liczba przykładów treningowych = {len(self.train_dataset)}")
        logger.info(f" Liczba epok = {self.args.num_train_epochs}")
        logger.info(f" Rozmiar batcha na urządzenie = {self.args.per_device_train_batch_size}")
        logger.info(f" Akumulacja gradientów kroków = {self.args.gradient_accumulation_steps}")
        effective_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        logger.info(f" Efektywny rozmiar batcha = {effective_batch_size}")

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=0, # Zmieniono z 4 na 0
            pin_memory=True if self.device.type == "cuda" else False,
            shuffle=True,
        )

        self.model.train()

        total_loss = 0.0
        global_step = 0 # Ten global_step nie jest faktycznie używany do logowania
        optimizer_step_count = 0

        for epoch in range(int(self.args.num_train_epochs)):
            epoch_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Debugowanie input_ids przed przekazaniem do modelu
                if batch['input_ids'].max().item() >= self.model.config.vocab_size:
                    problematic_indices = batch['input_ids'][batch['input_ids'] >= self.model.config.vocab_size]
                    logger.error(f"Input ID out of vocabulary range in batch! Max ID in batch: {batch['input_ids'].max().item()}, Vocab size: {self.model.config.vocab_size}")
                    logger.error(f"Problematic indices found: {problematic_indices}")
                    raise ValueError("Input IDs contain values outside of model's vocabulary range.")

                with torch.amp.autocast(device_type="cuda", enabled=self.args.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * self.args.gradient_accumulation_steps

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    optimizer_step_count += 1

                    if optimizer_step_count % self.args.logging_steps == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        logger.info(
                            f"Epoch: {epoch}, Step (optimizer): {optimizer_step_count}, Loss: {epoch_loss / (step + 1):.4f}, "
                            f"LR: {current_lr:.6f}"
                        )

                    if optimizer_step_count % self.args.eval_steps == 0 and self.eval_dataset is not None:
                        self.evaluate(optimizer_step_count)
                    
                    if optimizer_step_count % self.args.save_steps == 0:
                        self.save_model(output_dir=self.args.output_dir, step=optimizer_step_count)
            
            logger.info(f"Epoch {epoch} zakończona. Średni loss epoki: {epoch_loss / len(train_dataloader):.4f}")

        logger.info("***** Trening zakończony *****")
        self.save_model(output_dir=self.args.output_dir, final=True)

    def evaluate(self, global_step: int):
        """
        Przeprowadza ewaluację modelu na zbiorze walidacyjnym.
        """
        if self.eval_dataset is None:
            logger.info("Brak datasetu walidacyjnego. Pomijam ewaluację.")
            return

        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=0, # Zmieniono z 2 na 0
            pin_memory=True if self.device.type == "cuda" else False,
            shuffle=False,
        )

        self.model.eval()
        total_eval_loss = 0.0
        num_eval_steps = 0

        logger.info(f"***** Rozpoczynanie ewaluacji w kroku {global_step} *****")
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = self.model(**batch)
                
                loss = outputs.loss
                total_eval_loss += loss.item()
                num_eval_steps += 1
        
        avg_eval_loss = total_eval_loss / num_eval_steps
        perplexity = math.exp(avg_eval_loss)
        logger.info(f"***** Wyniki Ewaluacji w kroku {global_step} *****")
        logger.info(f" Eval Loss: {avg_eval_loss:.4f}")
        logger.info(f" Perplexity: {perplexity:.2f}")

        self.model.train()
        return {"eval_loss": avg_eval_loss, "perplexity": perplexity}


    def save_model(self, output_dir: str, step: Optional[int] = None, final: bool = False):
        """
        Zapisuje stan modelu, tokenizatora i konfiguracji.
        """
        save_path = output_dir
        if step is not None:
            save_path = os.path.join(output_dir, f"checkpoint-{step}")
        elif final:
            save_path = os.path.join(output_dir, "final")

        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path, safe_serialization=False)
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model zapisany w: {save_path}")

    def load_model(self, model_path: str):
        """
        Wczytuje model i tokenizator z podanej ścieżki.
        """
        logger.info(f"Wczytywanie modelu z: {model_path}")
        
        config = BlackholeConfig.from_pretrained(model_path)
        
        # Wczytanie tokenizatora przed modelem, aby uzyskać aktualny vocab_size
        if self.tokenizer: # Użyj istniejącego tokenizatora, jeśli dostępny
            self.tokenizer = BlackholeTokenizer.from_pretrained(model_path)
        else: # W przeciwnym razie stwórz nowy
            try:
                self.tokenizer = BlackholeTokenizer.from_pretrained(model_path)
            except Exception as e:
                logger.warning(f"Nie udało się wczytać tokenizatora z {model_path}. Kontynuowanie bez tokenizatora. Błąd: {e}")
                self.tokenizer = None # Zapewnij, że jest None, jeśli nie udało się wczytać

        # Upewnij się, że vocab_size w configu jest zgodny z tokenizatorem po wczytaniu
        if self.tokenizer and config.vocab_size != len(self.tokenizer):
            logger.warning(
                f"Niezgodność vocab_size! Konfiguracja modelu ({config.vocab_size}) "
                f"różni się od tokenizatora ({len(self.tokenizer)}). "
                "Aktualizuję config.vocab_size."
            )
            config.vocab_size = len(self.tokenizer)

        if config.architectures is not None:
            if "BlackholeForMaskedLM" in config.architectures:
                self.model = BlackholeForMaskedLM.from_pretrained(model_path, config=config)
            elif "BlackholeForSequenceClassification" in config.architectures:
                self.model = BlackholeForSequenceClassification.from_pretrained(model_path, config=config)
            else:
                raise ValueError(f"Nieznany typ architektury '{config.architectures}' w konfiguracji modelu. "
                                 f"Sprawdź plik config.json w {model_path}.")
        else:
            logger.warning(f"Brak atrybutu 'architectures' w config.json dla {model_path}. "
                            "Domyślnie ładowanie jako BlackholeForMaskedLM.")
            self.model = BlackholeForMaskedLM.from_pretrained(model_path, config=config)


        self.model.to(self.device)
        
        logger.info("Model wczytany pomyślnie.")
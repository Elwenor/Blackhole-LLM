# blackhole/nova_hugging_face/configuration_nova.py
from transformers import PretrainedConfig

class BlackholeConfig(PretrainedConfig):
    model_type = "blackhole"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        pad_token_id=0,
        num_token_id=5, # Example default, should be set by tokenizer
        numeric_feature_dims=None,
        numeric_embedding_fusion_type="gating",
        numeric_heavy_feature_freeze=False, # ADDED: Default for freezing heavy numeric features
        classifier_dropout=None, # For sequence classification
        num_labels=2, # For sequence classification
        problem_type=None, # For sequence classification
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_token_id = num_token_id
        self.numeric_feature_dims = numeric_feature_dims if numeric_feature_dims is not None else {}
        self.numeric_embedding_fusion_type = numeric_embedding_fusion_type
        self.numeric_heavy_feature_freeze = numeric_heavy_feature_freeze # ASSIGNED
        self.classifier_dropout = classifier_dropout
        self.num_labels = num_labels
        self.problem_type = problem_type
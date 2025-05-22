import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming blackhole.embedding is correctly installed and accessible
from blackhole.embedding import TokenEmbedding, NumberEmbedding

class ImprovedCrossEmbeddingSeq2SeqModel(nn.Module):
    """
    Model Encoder-Decoder łączący osadzenia tokenów i liczb za pomocą Transformerów.
    Zaprojektowany do zadań Question Answering z odpowiedziami zawierającymi tekst i liczby.
    """
    def __init__(self, vocab_size, token_dim, num_dim, hidden,
                 encoder_layers, decoder_layers, dropout, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden = hidden

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=token_dim)
        self.num_embedding = NumberEmbedding(input_dim=feature_dim, output_dim=num_dim)

        # Projections to common hidden dimension
        self.token_to_common_dim = nn.Linear(token_dim, hidden)
        self.num_to_common_dim = nn.Linear(num_dim, hidden)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Output Heads
        self.token_head = nn.Linear(hidden, vocab_size)
        self.num_head = nn.Sequential(
            nn.Linear(hidden, feature_dim),
            nn.Tanh() # Tanh to constrain values between -1 and 1
        )

    def forward(self, encoder_token_ids, encoder_numeric_features, encoder_attention_mask,
                decoder_token_ids, decoder_numeric_features_input, decoder_attention_mask):

        enc_token_emb = self.token_embedding(encoder_token_ids)
        enc_num_emb = self.num_embedding(encoder_numeric_features)

        enc_token_emb_proj = self.token_to_common_dim(enc_token_emb)
        enc_num_emb_proj = self.num_to_common_dim(enc_num_emb)

        encoder_input_emb = enc_token_emb_proj + enc_num_emb_proj

        # Invert attention mask for transformer (True means masked)
        encoder_output = self.transformer_encoder(encoder_input_emb, src_key_padding_mask=encoder_attention_mask)

        dec_token_emb = self.token_embedding(decoder_token_ids)
        dec_num_emb = self.num_embedding(decoder_numeric_features_input)

        dec_token_emb_proj = self.token_to_common_dim(dec_token_emb)
        dec_num_emb_proj = self.num_to_common_dim(dec_num_emb)

        decoder_input_emb = dec_token_emb_proj + dec_num_emb_proj

        tgt_seq_len = decoder_token_ids.size(1)
        # Create a mask that masks future tokens (auto-regression)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(decoder_token_ids.device)

        decoder_output = self.transformer_decoder(
            tgt=decoder_input_emb,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_attention_mask, # Invert attention mask for transformer
            memory_key_padding_mask=encoder_attention_mask # Invert attention mask for transformer
        )

        token_logits = self.token_head(decoder_output)
        num_feature_output = self.num_head(decoder_output)

        return token_logits, num_feature_output
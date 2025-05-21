import torch
import torch.nn as nn

class NovaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers, feature_dim, dropout=0.1):
        super().__init__()
        
        # --- Token and Numerical Embeddings ---
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # Assuming you have a dedicated numerical embedding layer
        # If not, you might process raw features directly or use a simple linear layer
        self.numerical_embedding_layer = nn.Linear(feature_dim, embedding_dim) # Example
        self.feature_dim = feature_dim # Store this for convenience, especially for padded_feat_row

        # --- Encoder ---
        # Define your encoder layers (e.g., TransformerEncoderLayer blocks)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Decoder ---
        # Define your decoder layers (e.g., TransformerDecoderLayer blocks)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- Output Layers ---
        # Output layer for token prediction (logits over vocabulary)
        self.token_output_layer = nn.Linear(embedding_dim, vocab_size)
        # Output layer for numerical feature prediction
        self.numerical_output_layer = nn.Linear(embedding_dim, feature_dim)
        
        # Positional encoding (if not handled by Transformer layers directly)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout) # Example
        
        # Layer normalization, dropout etc.
        self.dropout = nn.Dropout(dropout)


    def forward(self, encoder_token_ids, encoder_numeric_features, encoder_attention_mask,
                decoder_token_ids, decoder_numeric_features_input, decoder_attention_mask):
        # --- Encoder Input Processing ---
        # Combine token and numerical embeddings for encoder input
        token_emb = self.token_embedding(encoder_token_ids)
        numeric_emb = self.numerical_embedding_layer(encoder_numeric_features)
        
        # Simple combination, you might have a more complex fusion strategy
        encoder_input_embeddings = token_emb + numeric_emb 
        encoder_input_embeddings = self.positional_encoding(encoder_input_embeddings) # Apply positional encoding
        encoder_input_embeddings = self.dropout(encoder_input_embeddings)

        # Pass through encoder
        encoder_output = self.encoder(src=encoder_input_embeddings, src_key_padding_mask=encoder_attention_mask)

        # --- Decoder Input Processing ---
        # Combine token and numerical embeddings for decoder input
        decoder_token_emb = self.token_embedding(decoder_token_ids)
        decoder_numeric_emb = self.numerical_embedding_layer(decoder_numeric_features_input)

        decoder_input_embeddings = decoder_token_emb + decoder_numeric_emb
        decoder_input_embeddings = self.positional_encoding(decoder_input_embeddings) # Apply positional encoding
        decoder_input_embeddings = self.dropout(decoder_input_embeddings)

        # Pass through decoder
        # tgt_mask should be a causal mask for auto-regressive decoding
        # tgt_key_padding_mask for actual padding
        seq_len = decoder_input_embeddings.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=decoder_input_embeddings.device)
        
        decoder_output = self.decoder(
            tgt=decoder_input_embeddings,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_attention_mask,
            memory_key_padding_mask=encoder_attention_mask
        )

        # --- Output Predictions ---
        token_logits = self.token_output_layer(decoder_output)
        num_feature_output = self.numerical_output_layer(decoder_output)

        return token_logits, num_feature_output


# Example of Positional Encoding (if you need it, often included in Transformer implementations)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1) # Transpose pe to match (1, seq_len, embed_dim)
        return self.dropout(x)
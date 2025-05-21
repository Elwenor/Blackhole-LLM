import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from num2words import num2words
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blackhole.embedding import NumberEmbedding

# Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, output_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, output_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        return self.embedding(x)


class GatedCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context, mask=None):
        B, L, D = x.shape
        B, L_ctx, D = context.shape

        x_norm = self.norm(x)

        q = self.to_q(x_norm).view(B, L, self.heads, D//self.heads).transpose(1, 2)
        k = self.to_k(context).view(B, L_ctx, self.heads, D//self.heads).transpose(1, 2)
        v = self.to_v(context).view(B, L_ctx, self.heads, D//self.heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.heads, -1, L_ctx)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        gated = torch.sigmoid(self.gate(torch.cat([out, x], dim=-1)))
        fused = gated * out + (1 - gated) * x

        return x + self.to_out(fused)


class BilinearFusion(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(dim * 2, dim)

    def forward(self, a, b):
        bilinear_out = self.bilinear(a, b)
        combined = torch.cat([a, b], dim=-1)
        projected = self.project(combined)
        fused = bilinear_out + projected
        return self.norm(self.dropout(fused)) + a


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.net(self.norm(x))


class ImprovedCrossEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, token_dim=128, num_dim=128, hidden=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, token_dim)
        self.num_emb = NumberEmbedding(input_dim=128, output_dim=num_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, token_dim))

        self.proj_t = nn.Linear(token_dim, hidden)
        self.proj_n = nn.Linear(num_dim, hidden)

        self.att_layers = nn.ModuleList([])
        self.ff_layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.att_layers.append(nn.ModuleList([
                GatedCrossAttention(hidden, dropout=dropout),
                GatedCrossAttention(hidden, dropout=dropout)
            ]))
            self.ff_layers.append(FeedForward(hidden, hidden * 4, dropout))

        self.fuse = BilinearFusion(hidden, dropout)
        self.final_norm = nn.LayerNorm(hidden)

        self.token_pred = nn.Linear(hidden, vocab_size)
        self.num_pred = nn.Linear(hidden, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tok_ids, num_feats, attention_mask=None, return_emb=False):
        B, L = tok_ids.shape
        t_emb = self.token_emb(tok_ids)

        t_emb = t_emb + self.pos_emb[:, :L, :]
        t_emb = self.dropout(t_emb)

        t = self.proj_t(t_emb)
        n = self.proj_n(self.num_emb(num_feats))

        t2n_final = t
        n2t_final = n

        for att_pair, ff in zip(self.att_layers, self.ff_layers):
            att_tn, att_nt = att_pair

            t2n = att_tn(t, n, mask=attention_mask)
            n2t = att_nt(n, t, mask=attention_mask)

            t2n = ff(t2n)
            n2t = ff(n2t)

            t2n_final, n2t_final = t2n, n2t

        fused = self.fuse(t2n_final, n2t_final)
        fused = self.final_norm(fused)

        logits = self.token_pred(fused)
        num_out = self.num_pred(fused)

        if return_emb:
            return logits, t2n_final, n2t_final, num_out

        return logits, num_out
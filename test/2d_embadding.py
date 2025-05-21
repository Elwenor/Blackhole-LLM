import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
import numpy as np
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blackhole.tokenizer import tokenize, detokenize, summarize_tokens
from blackhole.embedding import TokenEmbedding, NumberEmbedding, number_embedding_features, prepare_inputs

text = """
During the trial, the compound concentration peaked at 1.23456789e+10 mol/L, but dropped to nearly zero (0.000000000045) within 3.14 seconds.
"""

# 1. Tokenizacja
tokens, number_map = tokenize(text)

# 2. Przygotowanie wejścia
token_ids, num_map, vocab = prepare_inputs(tokens, number_map)

# 3. Inicjalizacja embeddingów
token_emb_model = TokenEmbedding(vocab_size=len(vocab))
number_emb_model = NumberEmbedding()

# 4. Embedding tokenów
token_embeddings = token_emb_model(token_ids)  # [1, L, D_token]

# 5. Przygotowanie cech liczbowych
B, L = token_ids.shape
raw_feats = torch.zeros(B, L, 12)
for b in range(B):
    for i in range(L):
        if i in num_map:
            val, typ, raw = num_map[i]
            raw_feats[b, i] = number_embedding_features(val, typ)

# 6. Embedding liczb
number_embeddings = number_emb_model(raw_feats)  # [1, L, D_number]

# 7. Zamiana tensorów na numpy do PCA
token_emb_np = token_embeddings[0].detach().cpu().numpy()
number_emb_np = number_embeddings[0].detach().cpu().numpy()

# 8. PCA 2D (osobno dla token i number embeddingów)
pca_token = PCA(n_components=2)
token_2d = pca_token.fit_transform(token_emb_np)

pca_number = PCA(n_components=2)
number_2d = pca_number.fit_transform(number_emb_np)

# Przygotowanie figure i axes z czarnym tłem
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax in axes:
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

# Kolory i wielkości kropkowe - teraz większe
token_colors = ['cyan'] * L
number_colors = ['magenta'] * L
token_sizes = [100] * L
number_sizes = [100] * L

# Scattery z większymi kropkami i listami kolorów/wielkości
sc_token = axes[0].scatter(token_2d[:, 0], token_2d[:, 1], color=token_colors, s=token_sizes)
axes[0].set_title("Token Embeddings (PCA 2D)", color='white')
axes[0].grid(True, color='gray', linestyle='--', alpha=0.3)

sc_number = axes[1].scatter(number_2d[:, 0], number_2d[:, 1], color=number_colors, s=number_sizes)
axes[1].set_title("Number Embeddings (PCA 2D)", color='white')
axes[1].grid(True, color='gray', linestyle='--', alpha=0.3)

# Tooltips
annot_token = axes[0].annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                              bbox=dict(boxstyle="round", fc="w"),
                              arrowprops=dict(arrowstyle="->"))
annot_token.set_visible(False)

annot_number = axes[1].annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                               bbox=dict(boxstyle="round", fc="w"),
                               arrowprops=dict(arrowstyle="->"))
annot_number.set_visible(False)

def reset_highlights():
    sc_token.set_facecolor(['cyan'] * L)
    sc_token.set_sizes([100] * L)
    sc_number.set_facecolor(['magenta'] * L)
    sc_number.set_sizes([100] * L)

def highlight_related(idx, threshold=0.5):
    colors_token = ['cyan'] * L
    sizes_token = [100] * L
    colors_number = ['magenta'] * L
    sizes_number = [100] * L

    colors_token[idx] = 'yellow'
    sizes_token[idx] = 300
    colors_number[idx] = 'yellow'
    sizes_number[idx] = 300

    for i in range(L):
        corr_tn = np.corrcoef(token_emb_np[i], number_emb_np[idx])[0,1]
        corr_nt = np.corrcoef(token_emb_np[idx], number_emb_np[i])[0,1]
        if i != idx:
            if corr_tn > threshold:
                colors_token[i] = 'lime'
                sizes_token[i] = 200
            if corr_nt > threshold:
                colors_number[i] = 'lime'
                sizes_number[i] = 200

    sc_token.set_facecolor(colors_token)
    sc_token.set_sizes(sizes_token)
    sc_number.set_facecolor(colors_number)
    sc_number.set_sizes(sizes_number)

def update_annot_token(ind):
    idx = ind["ind"][0]
    pos = sc_token.get_offsets()[idx]
    annot_token.xy = pos
    
    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    ax = axes[0]
    inv = ax.transData.transform
    point_px = inv(pos)
    
    offset_x = 15 if point_px[0] < fig_width * 0.75 else -150
    offset_y = 15 if point_px[1] < fig_height * 0.75 else -60
    annot_token.set_position((offset_x, offset_y))
    
    token_text = tokens[idx]
    corrs = [np.corrcoef(token_emb_np[idx], number_emb_np[i])[0, 1] for i in range(L)]
    top_idxs = np.argsort(corrs)[-3:][::-1]
    top_corrs_str = "\n".join([f"Num idx {i}: {corrs[i]:.3f}" for i in top_idxs])

    annot_token.set_text(f"Token: {token_text}\nTop Number Corrs:\n{top_corrs_str}")
    annot_token.get_bbox_patch().set_alpha(0.9)

    highlight_related(idx)

def update_annot_number(ind):
    idx = ind["ind"][0]
    pos = sc_number.get_offsets()[idx]
    annot_number.xy = pos
    
    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    ax = axes[1]
    inv = ax.transData.transform
    point_px = inv(pos)
    
    offset_x = 15 if point_px[0] < fig_width * 0.75 else -150
    offset_y = 15 if point_px[1] < fig_height * 0.75 else -60
    annot_number.set_position((offset_x, offset_y))
    
    corrs = [np.corrcoef(token_emb_np[i], number_emb_np[idx])[0,1] for i in range(L)]
    top_idxs = np.argsort(corrs)[-3:][::-1]
    top_corrs_str = "\n".join([f"Token idx {i} ('{tokens[i]}'): {corrs[i]:.3f}" for i in top_idxs])

    annot_number.set_text(f"Number Embedding\nTop Token Corrs:\n{top_corrs_str}")
    annot_number.get_bbox_patch().set_alpha(0.9)

    highlight_related(idx)

def hover(event):
    vis_token = annot_token.get_visible()
    vis_number = annot_number.get_visible()
    if event.inaxes == axes[0]:
        cont, ind = sc_token.contains(event)
        if cont:
            update_annot_token(ind)
            annot_token.set_visible(True)
            if vis_number:
                annot_number.set_visible(False)
            fig.canvas.draw_idle()
        else:
            if vis_token:
                annot_token.set_visible(False)
                reset_highlights()
                fig.canvas.draw_idle()
    elif event.inaxes == axes[1]:
        cont, ind = sc_number.contains(event)
        if cont:
            update_annot_number(ind)
            annot_number.set_visible(True)
            if vis_token:
                annot_token.set_visible(False)
            fig.canvas.draw_idle()
        else:
            if vis_number:
                annot_number.set_visible(False)
                reset_highlights()
                fig.canvas.draw_idle()
    else:
        if vis_token or vis_number:
            annot_token.set_visible(False)
            annot_number.set_visible(False)
            reset_highlights()
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

# Na końcu wypisz 20 losowych korelacji token-number
def explain_corr(corr):
    if corr > 0.7:
        return "Silna korelacja — podobne wzorce"
    elif corr > 0.3:
        return "Średnia korelacja"
    elif corr > 0:
        return "Słaba korelacja"
    elif corr > -0.3:
        return "Brak korelacji"
    elif corr > -0.7:
        return "NEGATYWNA korelacja"
    else:
        return "NEGATYWNA korelacja — przeciwstawne wzorce"

print("Orginalny text: " + text)
print("Przykładowe 20 korelacji Token-Number embeddings (losowe pary):\n")
print(f"{'Token idx':<10} {'Token':<15} {'Number idx':<12} {'Korelacja':<10} {'Interpretacja'}")
print("-" * 65)

for _ in range(20):
    t_idx = random.randint(0, L-1)
    n_idx = random.randint(0, L-1)
    corr = np.corrcoef(token_emb_np[t_idx], number_emb_np[n_idx])[0,1]
    
    explanation = explain_corr(corr)
    print(f"{t_idx:<10} '{tokens[t_idx]:<13}' {n_idx:<12} {corr:<10.3f} {explanation}")


plt.show()

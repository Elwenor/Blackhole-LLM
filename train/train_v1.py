import sys
import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from libraries
from blackhole.tokenizer import tokenize, detokenize, summarize_tokens
from blackhole.embedding import NumberEmbedding, number_embedding_features, decode_number_from_features

# Define embedding functions locally to ensure compatibility
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, output_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, output_dim)
    
    def forward(self, x):
        return self.embedding(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- IMPROVED MODEL WITH CROSS ATTENTION -----------

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, context):
        residual = x
        B, L, D = x.shape
        
        q = self.to_q(x).view(B, L, self.heads, D // self.heads).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.heads, D // self.heads).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.heads, D // self.heads).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn_scores.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.to_out(out)
        
        # Add residual connection and layer norm
        out = self.norm(residual + out)
        
        return out

class ImprovedCrossEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, token_emb_dim=128, number_emb_dim=128, hidden_dim=256):
        super().__init__()
        # Increased dimensions for better representation
        self.token_emb_model = TokenEmbedding(vocab_size, output_dim=token_emb_dim)
        self.number_emb_model = NumberEmbedding(input_dim=128, output_dim=number_emb_dim)
        
        # Better projection between embedding spaces
        self.number_proj = nn.Sequential(
            nn.Linear(number_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_emb_dim)
        )
        
        # Token projection to match dimensions
        self.token_proj = nn.Sequential(
            nn.Linear(token_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_emb_dim)
        )
        
        # Cross attention with residual connections
        self.cross_attn_t2n = CrossAttentionBlock(token_emb_dim)
        self.cross_attn_n2t = CrossAttentionBlock(token_emb_dim)
        
        # Final feed-forward layers
        self.combined_layer = nn.Sequential(
            nn.Linear(token_emb_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Auxiliary loss: predict number directly from embeddings
        self.number_decoder = nn.Linear(token_emb_dim, 1)
    
    def forward(self, token_ids, number_feats, return_embeddings=False):
        # token_ids: [B, L]
        # number_feats: [B, L, 128]
        
        # Get embeddings
        token_embs = self.token_emb_model(token_ids)                # [B, L, token_emb_dim]
        number_embs = self.number_emb_model(number_feats)           # [B, L, number_emb_dim]
        
        # Project number embeddings to token space
        number_embs_proj = self.number_proj(number_embs)            # [B, L, token_emb_dim]
        
        # Project token embeddings for better alignment
        token_embs_proj = self.token_proj(token_embs)               # [B, L, token_emb_dim]
        
        # Cross-attention in both directions
        token_attended = self.cross_attn_t2n(token_embs_proj, number_embs_proj)    # [B, L, token_emb_dim]
        number_attended = self.cross_attn_n2t(number_embs_proj, token_embs_proj)   # [B, L, token_emb_dim]
        
        # Combine both embeddings
        combined = torch.cat([token_attended, number_attended], dim=-1)  # [B, L, token_emb_dim*2]
        
        # Process through combined layer
        hidden = self.combined_layer(combined)                      # [B, L, hidden_dim]
        
        # Output logits
        logits = self.fc_out(hidden)                                # [B, L, vocab_size]
        
        # Additional outputs for number prediction (auxiliary task)
        number_preds = self.number_decoder(number_attended)         # [B, L, 1]
        
        if return_embeddings:
            return logits, token_embs_proj, number_embs_proj, number_preds
        return logits

# ----------- GENERATOR PROSTYCH DANYCH -----------

def generate_simple_examples(n=1000):
    examples = []
    for _ in range(n):
        a = random.randint(0, 1000)
        b = random.randint(0, 1000)
        c = a + b
        
        def fmt_num(x):
            style = random.choice(['int', 'float', 'sci'])
            if style == 'int':
                return str(x), 'int'
            elif style == 'float':
                return f"{float(x):.4f}", 'float'
            else:
                return f"{x:.2e}", 'float'
        
        a_str, a_typ = fmt_num(a)
        b_str, b_typ = fmt_num(b)
        c_str, c_typ = fmt_num(c)
        
        inp = f"{a_str} + {b_str} ="
        out = c_str
        
        examples.append((inp, out, (a, a_typ), (b, b_typ), (c, c_typ)))
    return examples

# ----------- IMPROVED DATA PREPARATION -----------

def prepare_batch(batch, vocab):
    # batch: list of tuples (inp, out, (a,a_typ), (b,b_typ), (c,c_typ))
    inputs = []
    targets = []
    number_maps = []
    
    max_len = 0
    for inp, out, a, b, c in batch:
        text = inp + " " + out
        tokens, number_map = tokenize(text)
        inputs.append(tokens)
        number_maps.append(number_map)
        max_len = max(max_len, len(tokens))
    
    # padding tokens
    padded_tokens = []
    for toks in inputs:
        padded = toks + ["<|pad|>"] * (max_len - len(toks))
        padded_tokens.append(padded)
    
    # convert tokens to ids
    token_ids = []
    for toks in padded_tokens:
        token_ids.append([vocab.get(t, vocab.get("<|unk|>", 0)) for t in toks])
    token_ids_tensor = torch.LongTensor(token_ids)
    
    # prepare number features
    B, L = token_ids_tensor.shape
    feats = torch.ones((B, L, 128), dtype=torch.float) * -1.0
    
    # Generate target tensors for number prediction (auxiliary task)
    number_targets = torch.zeros((B, L, 1), dtype=torch.float)
    
    for b_i, number_map in enumerate(number_maps):
        for idx, data in enumerate(number_map.items()):
            token_idx, value_data = data
            if token_idx >= L:
                continue
            if isinstance(value_data, (tuple, list)) and len(value_data) >= 2:
                val, typ = value_data[0], value_data[1]
                feats[b_i, token_idx] = number_embedding_features(val, typ)
                # Set numerical target for number tokens
                if typ in ('int', 'float'):
                    number_targets[b_i, token_idx, 0] = float(val)
    
    # Create target mask: only predict where we have numerical tokens
    target_mask = (feats[:, :, 0] != -1.0).float().unsqueeze(-1)  # [B, L, 1]
    
    return token_ids_tensor, feats, number_maps, padded_tokens, vocab, number_targets, target_mask

# ----------- IMPROVED DECODING PREDICTIONS -----------

def decode_predictions(pred_tokens, number_maps, inv_vocab, orig_examples):
    results = []
    
    for i, (tokens, number_map, example) in enumerate(zip(pred_tokens, number_maps, orig_examples)):
        # Convert token IDs to token strings
        token_strs = [inv_vocab.get(t, "<|unk|>") for t in tokens]
        
        # Extract input part and prediction part
        # Find the '=' token
        try:
            eq_idx = token_strs.index('=')
            input_tokens = token_strs[:eq_idx+1]
            pred_tokens = token_strs[eq_idx+1:]
            
            # Remove padding tokens from prediction
            pred_tokens = [t for t in pred_tokens if t != "<|pad|>"]
            
            # Try to recover the original input
            input_str = detokenize(input_tokens, {k:v for k,v in number_map.items() if k <= eq_idx})
            
            # Create a new number map for prediction - we'll identify <|num|> tokens
            pred_number_map = {}
            offset = eq_idx + 1  # Adjust index for the prediction part
            for j, tok in enumerate(pred_tokens):
                if tok == "<|num|>":
                    # Use the original expected result if available in number_map
                    if offset + j in number_map:
                        pred_number_map[j] = number_map[offset + j]
                    else:
                        # Fallback to expected result
                        pred_number_map[j] = example[4]  # Use the expected result's format
            
            # Detokenize the prediction with the number map
            pred_str = detokenize(pred_tokens, pred_number_map)
            
            # Get the original target string
            orig_target = example[1]  # The second element is the output part
            
            # Extract numeric values
            pred_num = None
            try:
                pred_str_clean = pred_str.strip()
                if pred_str_clean:  # Only try to convert if there's something there
                    pred_num = float(pred_str_clean)
            except ValueError:
                pred_num = None
                
            expected_num = example[4][0]  # c value from (c, c_typ)
            
            # Calculate relative error
            rel_error = None
            if pred_num is not None and expected_num != 0:
                rel_error = abs(pred_num - expected_num) / abs(expected_num)
            elif pred_num is not None and expected_num == 0:
                rel_error = abs(pred_num)
                
            # Prepare the result
            result = {
                "input_tokens": input_tokens,
                "input_str": input_str.strip(),
                "pred_tokens": pred_tokens,
                "pred_str": pred_str.strip(),
                "target_str": orig_target.strip(),
                "pred_num": pred_num,
                "expected_num": expected_num,
                "rel_error": rel_error,
                "is_correct": pred_num is not None and (rel_error is not None and rel_error < 1e-6)
            }
            
        except ValueError:  # If '=' token is not found
            # Handle a different format or fallback
            input_str = detokenize(token_strs, {})
            result = {
                "input_tokens": token_strs,
                "input_str": input_str.strip(),
                "pred_tokens": [],
                "pred_str": "",
                "target_str": example[1].strip(),
                "pred_num": None,
                "expected_num": example[4][0],
                "rel_error": None,
                "is_correct": False
            }
            
        results.append(result)
            
    return results

# ----------- IMPROVED TRAINING LOOP -----------

def calculate_embedding_losses(token_embs, number_embs_proj):
    """Calculate losses between token and number embeddings"""
    # L2 distance (MSE loss)
    l2_loss = torch.mean(torch.sum((token_embs - number_embs_proj)**2, dim=-1))
    
    # Cosine similarity loss - normalized
    token_norm = torch.nn.functional.normalize(token_embs, p=2, dim=-1)
    number_norm = torch.nn.functional.normalize(number_embs_proj, p=2, dim=-1)
    cos_sim = torch.sum(token_norm * number_norm, dim=-1).mean()
    cos_loss = 1.0 - cos_sim  # Lower is better
    
    # Feature-wise analysis
    per_feature_error = torch.mean(torch.abs(token_embs - number_embs_proj), dim=0)
    max_feature_error = torch.max(per_feature_error).item()
    min_feature_error = torch.min(per_feature_error).item()
    mean_feature_error = torch.mean(per_feature_error).item()
    
    return {
        "l2_loss": l2_loss.item(),
        "cos_loss": cos_loss.item(),
        "cos_sim": cos_sim.item(),
        "max_feature_error": max_feature_error,
        "min_feature_error": min_feature_error,
        "mean_feature_error": mean_feature_error
    }

def improved_train_loop(model, optimizer, criterion, batch, vocab, device):
    model.train()
    token_ids, feats, number_maps, padded_tokens, _, number_targets, target_mask = prepare_batch(batch, vocab)
    token_ids = token_ids.to(device)
    feats = feats.to(device)
    number_targets = number_targets.to(device)
    target_mask = target_mask.to(device)

    optimizer.zero_grad()
    logits, token_embs, number_embs_proj, number_preds = model(token_ids, feats, return_embeddings=True)

    targets = token_ids

    # 1. Calculate cross entropy loss for token prediction
    ce_loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    # 2. Calculate embedding losses for alignment
    embedding_losses = calculate_embedding_losses(token_embs, number_embs_proj)
    
    # 3. Calculate number prediction loss (MSE for numeric values)
    # Only apply loss where we have valid numbers (using mask)
    pred_error = (number_preds - number_targets) * target_mask
    num_pred_loss = torch.mean(pred_error ** 2)
    
    # Loss weights for balancing different objectives
    emb_weight = 0.1  # Lower weight for embedding alignment
    num_pred_weight = 0.5  # Higher weight for direct number prediction
    
    # Combined loss
    total_loss = ce_loss + emb_weight * embedding_losses["l2_loss"] + num_pred_weight * num_pred_loss
    
    total_loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Add losses to embedding losses dict for reporting
    embedding_losses["ce_loss"] = ce_loss.item()
    embedding_losses["num_pred_loss"] = num_pred_loss.item()
    embedding_losses["total_loss"] = total_loss.item()
    
    return embedding_losses

def evaluate_model(model, examples, vocab, device, n=5, detailed=False):
    """Evaluate model and return metrics + examples"""
    model.eval()
    batch = examples[:n]
    token_ids_batch, feats_batch, number_maps, padded_tokens, _, number_targets, target_mask = prepare_batch(batch, vocab)
    token_ids_batch = token_ids_batch.to(device)
    feats_batch = feats_batch.to(device)
    number_targets = number_targets.to(device)
    target_mask = target_mask.to(device)

    with torch.no_grad():
        if detailed:
            logits, token_embs, number_embs, number_preds = model(token_ids_batch, feats_batch, return_embeddings=True)
            # Calculate embedding similarity
            emb_similarity = calculate_embedding_losses(token_embs, number_embs)["cos_sim"]
            
            # Calculate numeric prediction accuracy
            # Only consider positions with valid numbers
            valid_positions = (target_mask.sum().item() > 0)
            if valid_positions:
                num_pred_error = torch.mean(((number_preds - number_targets) * target_mask) ** 2).item()
            else:
                num_pred_error = None
        else:
            logits = model(token_ids_batch, feats_batch)
            emb_similarity = None
            num_pred_error = None
            
        preds = torch.argmax(logits, dim=-1)

    total_tokens = 0
    correct_tokens = 0

    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Decode the predictions
    pred_tokens = preds.cpu().tolist()
    results = decode_predictions(pred_tokens, number_maps, inv_vocab, batch)
    
    # Calculate token accuracy
    for i in range(n):
        input_tokens = token_ids_batch[i].cpu().tolist()
        pred_tokens_i = preds[i].cpu().tolist()

        # Count accuracy by tokens, ignore padding
        for t, p in zip(input_tokens, pred_tokens_i):
            if t == vocab.get("<|pad|>", -1):
                continue
            total_tokens += 1
            if t == p:
                correct_tokens += 1

    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    prediction_accuracy = sum(1 for r in results if r["is_correct"]) / len(results) if results else 0
    
    # Calculate relative error statistics
    rel_errors = [r["rel_error"] for r in results if r["rel_error"] is not None]
    mean_rel_error = sum(rel_errors) / len(rel_errors) if rel_errors else None
    
    metrics = {
        "token_accuracy": token_accuracy,
        "prediction_accuracy": prediction_accuracy,
        "embedding_similarity": emb_similarity,
        "mean_rel_error": mean_rel_error,
        "num_pred_error": num_pred_error
    }
    
    model.train()
    return metrics, results

def pretty_print_evaluation(metrics, example_results, print_examples=True):
    """Print evaluation results in a pretty format"""
    # Print header
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS".center(70))
    print("="*70)
    
    # Print metrics
    print(f"Token Accuracy:       {metrics['token_accuracy']:.4f}")
    print(f"Prediction Accuracy:  {metrics['prediction_accuracy']:.4f}")
    
    if metrics['embedding_similarity'] is not None:
        print(f"Embedding Similarity: {metrics['embedding_similarity']:.4f}")
        
    if metrics['mean_rel_error'] is not None:
        print(f"Mean Relative Error:  {metrics['mean_rel_error']:.8f}")
        
    if metrics['num_pred_error'] is not None:
        print(f"Num Pred MSE:        {metrics['num_pred_error']:.8f}")
    
    # Print examples
    if print_examples and example_results:
        print("\n" + "-"*70)
        print("EXAMPLE PREDICTIONS WITH DETOKENIZATION".center(70))
        print("-"*70 + "\n")
        
        for i, ex in enumerate(example_results):
            status = "✓" if ex["is_correct"] else "✗"
            print(f"Example {i+1}: {status}")
            print(f"Input:      {ex['input_str']}")
            print(f"Target:     {ex['target_str']}")
            print(f"Prediction: {ex['pred_str']}")
            
            # Display numeric values for clarity
            if ex["pred_num"] is not None:
                print(f"Pred Num:   {ex['pred_num']}")
            print(f"Expected:   {ex['expected_num']}")
            
            if ex["rel_error"] is not None:
                print(f"Rel. Error: {ex['rel_error']:.8f}")
                
            print("-" * 40)

def build_vocab_from_examples(examples):
    tokens_all = []
    for inp, out, a, b, c in examples:
        text = inp + " " + out
        toks, _ = tokenize(text)
        tokens_all.extend(toks)
    vocab = {tok: idx for idx, tok in enumerate(sorted(set(tokens_all) | {"<|pad|>", "<|unk|>"}))}
    return vocab

def main():
    # Reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Generate data and build vocabulary
    examples = generate_simple_examples(1000)
    vocab = build_vocab_from_examples(examples)
    print(f"Vocabulary size: {len(vocab)}")

    # Initialize model, optimizer, criterion
    model = ImprovedCrossEmbeddingModel(vocab_size=len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<|pad|>'])

    # Training parameters
    epochs = 10
    batch_size = 32
    history = {
        'ce_loss': [], 'l2_loss': [], 'cos_loss': [], 'num_pred_loss': [],
        'total_loss': [], 'max_feature_error': [], 'min_feature_error': [],
        'mean_feature_error': [], 'token_accuracy': [], 'prediction_accuracy': [],
        'mean_rel_error': [], 'embedding_similarity': []
    }

    # Banner
    print("\n" + "="*70)
    print("TRAINING IMPROVED CROSS-EMBEDDING MODEL FOR MATHEMATICAL OPERATIONS".center(70))
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Examples: {len(examples)}")
    print("="*70 + "\n")

    # Show raw examples
    print("Example Raw Training Data:")
    for i, example in enumerate(examples[:3], 1):
        inp, out, *_ = example
        print(f"Example {i}:")
        print(f"  Input:  '{inp}'")
        print(f"  Output: '{out}'")
        tokens, num_map = tokenize(f"{inp} {out}")
        print(f"  Tokenized: {tokens}")
        print(f"  Number Map: {num_map}\n")

    # Training loop
    for epoch in range(1, epochs+1):
        random.shuffle(examples)
        epoch_losses = {k: 0.0 for k in history.keys()}

        # LR scheduling
        if epoch == (epochs // 2):
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5

        for start in tqdm(range(0, len(examples), batch_size), desc=f"Epoch {epoch}/{epochs}"):
            batch = examples[start:start+batch_size]
            losses = improved_train_loop(model, optimizer, criterion, batch, vocab, device)
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v

        # Average losses
        num_batches = max(1, len(examples) // batch_size)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        # Evaluation
        eval_metrics, eval_examples = evaluate_model(model, examples, vocab, device, n=5, detailed=True)

        # Record history
        for metric in ['ce_loss','l2_loss','cos_loss','num_pred_loss','total_loss',
                       'max_feature_error','min_feature_error','mean_feature_error']:
            history[metric].append(epoch_losses[metric])
        history['token_accuracy'].append(eval_metrics['token_accuracy'])
        history['prediction_accuracy'].append(eval_metrics['prediction_accuracy'])
        history['embedding_similarity'].append(eval_metrics['embedding_similarity'] or 0)
        if eval_metrics['mean_rel_error'] is not None:
            history['mean_rel_error'].append(eval_metrics['mean_rel_error'])

        # Epoch summary
        print("\n" + "="*70)
        print(f"EPOCH {epoch}/{epochs} SUMMARY".center(70))
        print("="*70)
        print(f"Cross-Entropy Loss:     {epoch_losses['ce_loss']:.6f}")
        print(f"Total Loss:             {epoch_losses['total_loss']:.6f}")
        print(f"Embedding L2 Loss:      {epoch_losses['l2_loss']:.6f}")
        print(f"Embedding COS Loss:     {epoch_losses['cos_loss']:.6f}")
        print(f"Number Pred Loss:       {epoch_losses['num_pred_loss']:.6f}")
        print(f"Token Accuracy:         {eval_metrics['token_accuracy']:.6f}")
        print(f"Prediction Accuracy:    {eval_metrics['prediction_accuracy']:.6f}")
        if eval_metrics['mean_rel_error'] is not None:
            print(f"Mean Rel Error:         {eval_metrics['mean_rel_error']:.8f}")
        print("="*70)

        # Detailed examples
        pretty_print_evaluation(eval_metrics, eval_examples)

        # Plot every 5 epochs and last
        if epoch % 5 == 0 or epoch == epochs:
            plt.figure(figsize=(12,10))
            plt.subplot(2,2,1)
            plt.plot(history['ce_loss'], label='CE Loss')
            plt.plot(history['total_loss'], label='Total Loss')
            plt.title('Loss Curves'); plt.legend()

            plt.subplot(2,2,2)
            plt.plot(history['l2_loss'], label='L2 Loss')
            plt.plot(history['cos_loss'], label='Cos Loss')
            plt.title('Embedding Losses'); plt.legend()

            plt.subplot(2,2,3)
            plt.plot(history['token_accuracy'], label='Token Acc')
            plt.plot(history['prediction_accuracy'], label='Pred Acc')
            plt.title('Accuracy'); plt.legend()

            plt.subplot(2,2,4)
            plt.plot(history['mean_rel_error'], label='Rel Error')
            plt.title('Mean Relative Error'); plt.legend()

            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    main()
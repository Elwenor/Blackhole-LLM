import sys
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from num2words import num2words
import matplotlib.pyplot as plt

# Safe __file__ usage with proper error handling
try:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
except NameError:
    base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, base_dir)

# Set seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Imports (assuming these modules exist in the project)
# These imports might not be available, so we'll define fallback implementations
try:
    from blackhole.tokenizer import tokenize, detokenize
except ImportError:
    # Fallback tokenizer implementation
    def tokenize(text):
        """Simple tokenizer that splits on spaces and punctuation"""
        # Replace punctuation with spaces around them for splitting
        for char in ".,!?;:()[]{}+-*/=":
            text = text.replace(char, f" {char} ")
        tokens = text.split()
        # Simple mapping - each token maps to its position
        mapping = {i: i for i in range(len(tokens))}
        return tokens, mapping
    
    def detokenize(tokens, mapping=None):
        """Simple detokenizer that joins tokens with spaces"""
        return " ".join(tokens)

    class ImprovedCrossEmbeddingModel(nn.Module):
        """
        A simplified model that combines token and numeric embeddings.
        """
        def __init__(self, vocab_size, token_dim=128, num_dim=128, hidden=256, num_layers=3, dropout=0.2, feature_dim=2):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, token_dim)
            self.num_projection = nn.Linear(feature_dim, num_dim)  # Adjust to actual feature dimension
            
            # Token to number and number to token projections
            self.t2n = nn.Linear(token_dim, num_dim)
            self.n2t = nn.Linear(num_dim, token_dim)
            
            # Transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=8,
                dim_feedforward=hidden,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Output heads
            self.token_head = nn.Linear(token_dim, vocab_size)
            self.num_head = nn.Linear(token_dim, 1)
            
        def forward(self, token_ids, features, attention_mask=None, return_emb=False):
            """
            Forward pass through the model.
            
            Args:
                token_ids: Tensor of token IDs [batch_size, seq_len]
                features: Tensor of numeric features [batch_size, seq_len, feature_dim]
                attention_mask: Mask for attention [batch_size, seq_len]
                return_emb: Whether to return embeddings
                
            Returns:
                Tuple of outputs (token logits, numeric output, embeddings if return_emb=True)
            """
            # Token embeddings
            token_emb = self.token_embedding(token_ids)
            
            # Process features - handle dimensionality properly
            try:
                num_emb = self.num_projection(features)
            except RuntimeError:
                # If there's a dimensionality issue, create empty embeddings of the right shape
                print(f"Feature shape issue: {features.shape}. Creating zero embeddings instead.")
                num_emb = torch.zeros(
                    (features.size(0), features.size(1), self.num_projection.out_features),
                    device=features.device
                )
            
            # Token-to-number and number-to-token projections
            t2n_emb = self.t2n(token_emb)
            n2t_emb = self.n2t(num_emb)
            
            # Combine embeddings
            combined_emb = token_emb + n2t_emb
            
            # Create padding mask for transformer (convert attention_mask from 1=attend to 0=attend)
            if attention_mask is not None:
                padding_mask = (attention_mask == 0)
            else:
                padding_mask = None
            
            # Pass through transformer
            transformer_out = self.transformer(combined_emb, src_key_padding_mask=padding_mask)
            
            # Token classification head
            token_logits = self.token_head(transformer_out)
            
            # Numeric prediction head
            num_out = self.num_head(transformer_out)
            
            if return_emb:
                return token_logits, t2n_emb, n2t_emb, num_out
            else:
                return token_logits, num_out

def prepare_batch(batch, vocab):
    """
    Prepare a batch of examples for model training or evaluation.
    
    Args:
        batch: List of tuples (text, target, a_info, b_info, c_info)
        vocab: Dictionary mapping tokens to indices
        
    Returns:
        Tuple of tensors and mappings needed for model input
    """
    token_ids, features, attention_masks, numeric_targets, maps = [], [], [], [], []

    for text, target, a_info, b_info, c_info in batch:
        # Tokenize text
        tokens, mapping = tokenize(text)
        
        # Convert tokens to IDs
        ids = [vocab.get(tok, vocab.get('<|unk|>', 0)) for tok in tokens]
        
        # Create attention mask (1 for real tokens)
        attn = [1] * len(ids)
        
        # Extract features using our helper function
        feat = extract_features(tokens, [a_info, b_info])
        
        # Store values
        token_ids.append(ids)
        features.append(feat)
        attention_masks.append(attn)
        numeric_targets.append([float(c_info[0])])
        maps.append(mapping)

    # Maximum sequence length for padding
    max_len = max(len(x) for x in token_ids)
    pad_token_id = vocab.get('<|pad|>', 0)

    # Pad all sequences to the same length
    padded_token_ids = []
    padded_features = []
    padded_attention_masks = []
    
    for ids, feat, mask in zip(token_ids, features, attention_masks):
        # Pad token ids
        padded_ids = ids + [pad_token_id] * (max_len - len(ids))
        padded_token_ids.append(padded_ids)
        
        # Pad features
        padded_feat = feat + [[0.0] * len(feat[0])] * (max_len - len(feat)) if feat else [[0.0]] * max_len
        padded_features.append(padded_feat)
        
        # Pad attention mask
        padded_mask = mask + [0] * (max_len - len(mask))
        padded_attention_masks.append(padded_mask)
    
    # Convert to tensors
    token_ids = torch.tensor(padded_token_ids)
    features = torch.tensor(padded_features, dtype=torch.float32)
    attention_masks = torch.tensor(padded_attention_masks)
    numeric_targets = torch.tensor(numeric_targets, dtype=torch.float32)

    return token_ids, features, attention_masks, numeric_targets, maps

    return token_ids, features, attention_masks, numeric_targets, maps

def generate_improved_examples(n=1000, max_num=10000):
    """
    Generate examples for training and evaluation.
    
    Args:
        n: Number of examples to generate
        max_num: Maximum number to use in examples
        
    Returns:
        List of tuples (input, target, a_info, b_info, c_info)
    """
    templates = [
        "{a} + {b} =", "What is {a} plus {b}?", "Add {a} and {b}",
        "Compute sum of {a} and {b}", "The sum of {a} and {b} is",
        "{a} added to {b} equals", "Calculate {a} + {b}", "If you add {a} and {b}, you get",
        "What do you get when you add {a} and {b}?", "{a} plus {b} equals"
    ]

    def fmt(x, style=None):
        """Format a number as words or digits based on style"""
        if style is None:
            style = random.choice(['words', 'digits', 'digits'])
        return (num2words(x, lang='en'), 'words') if style == 'words' else (str(x), 'digits')

    samples = []
    for _ in range(n):
        # Generate numbers - 70% chance of smaller numbers, 30% chance of larger numbers
        a, b = (random.randint(0, 100), random.randint(0, 100)) if random.random() < 0.7 else (random.randint(0, max_num), random.randint(0, max_num))
        c = a + b
        
        # Format numbers as words or digits
        a_s, a_type = fmt(a, random.choice(['digits', 'digits', 'words']))
        b_s, b_type = fmt(b, random.choice(['digits', 'digits', 'words']))
        c_s, c_type = fmt(c, 'digits')
        
        # Create input text using a template
        inp = random.choice(templates).format(a=a_s, b=b_s)
        samples.append((inp, c_s, (a, a_type), (b, b_type), (c, c_type)))
    
    return samples

def build_vocab_from_examples(examples):
    """
    Build vocabulary from examples.
    
    Args:
        examples: List of example tuples
        
    Returns:
        Dictionary mapping tokens to indices
    """
    tokens_all = []
    for inp, out, *_ in examples:
        toks, _ = tokenize(f"{inp} {out}")
        tokens_all.extend(toks)
    
    # Add special tokens
    special_tokens = {'<|pad|>', '<|unk|>', '<|bos|>', '<|eos|>'}
    unique_tokens = set(tokens_all) | special_tokens
    
    # Create vocabulary
    vocab = {tok: i for i, tok in enumerate(sorted(unique_tokens))}
    return vocab

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, ignore_index=0):
    """
    Compute focal loss for classification.
    
    Args:
        logits: Predicted logits
        targets: Target indices
        alpha: Weighting factor
        gamma: Focusing parameter
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Focal loss value
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()

def cosine_similarity_loss(a, b):
    """
    Compute cosine similarity loss between embedding spaces.
    
    Args:
        a, b: Embedding tensors
        
    Returns:
        Loss value
    """
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    sim = torch.matmul(a_norm, b_norm.transpose(-2, -1))
    return 1 - sim.mean()

def huber_loss(pred, target, delta=1.0):
    """
    Compute Huber loss for regression.
    
    Args:
        pred: Predictions
        target: Targets
        delta: Threshold parameter
        
    Returns:
        Huber loss value
    """
    loss = torch.abs(pred - target)
    return torch.where(loss < delta, 0.5 * loss**2, delta * (loss - 0.5 * delta)).mean()

def train_step(model, optimizer, scheduler, batch, vocab, device, clip_grad=1.0):
    """
    Perform one training step.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        batch: Batch of examples
        vocab: Vocabulary
        device: Device to use
        clip_grad: Gradient clipping value
        
    Returns:
        Dictionary of loss values
    """
    model.train()
    
    # Prepare batch and move to device
    tok_ids, feats, attention_mask, num_targets, _ = prepare_batch(batch, vocab)
    tok_ids, feats, attention_mask, num_targets = [x.to(device) for x in (tok_ids, feats, attention_mask, num_targets)]
    
    # Forward pass
    optimizer.zero_grad()
    logits, t2n, n2t, num_out = model(tok_ids, feats, attention_mask, return_emb=True)
    
    # Calculate losses
    pad_id = vocab.get('<|pad|>', 0)
    ce = focal_loss(logits.view(-1, logits.size(-1)), tok_ids.view(-1), ignore_index=pad_id)
    align = cosine_similarity_loss(t2n, n2t)
    
    # Handle numeric loss
    mask = (num_targets != 0).float()
    num_values_count = mask.sum().item()
    num_loss = huber_loss(num_out * mask, num_targets * mask) if num_values_count > 0 else torch.tensor(0.0).to(device)
    
    # Combine losses with weights
    loss = ce + 0.15 * align + 0.5 * num_loss
    
    # Backward pass and optimization
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    
    return {
        'ce': ce.item(), 
        'align': align.item(), 
        'num_loss': num_loss.item(), 
        'total': loss.item(), 
        'num_values': num_values_count
    }

def evaluate(model, examples, vocab, device):
    """
    Evaluate model on examples.
    
    Args:
        model: Model to evaluate
        examples: List of examples
        vocab: Vocabulary
        device: Device to use
        
    Returns:
        List of predicted answers
    """
    model.eval()
    results = []
    batch_size = 32
    
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        tok_ids, feats, attention_mask, _, maps = prepare_batch(batch, vocab)
        
        with torch.no_grad():
            logits, num_out = model(tok_ids.to(device), feats.to(device), attention_mask.to(device))
        
        pred_ids = logits.argmax(-1).cpu().tolist()
        inv_vocab = {v: k for k, v in vocab.items()}
        special_tokens = {'<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>'}
        
        for j, (seq, num_map) in enumerate(zip(pred_ids, maps)):
            # Convert token IDs to tokens, filtering out special tokens
            tokens = [inv_vocab.get(i, '<|unk|>') for i in seq 
                     if i in inv_vocab and inv_vocab[i] not in special_tokens]
            
            # Try to find answer after '=' or '?', otherwise take last few tokens
            if '=' in tokens:
                idx = tokens.index('=') + 1
            elif '?' in tokens:
                idx = tokens.index('?') + 1
            else:
                idx = max(0, len(tokens) - 5)
            
            answer_tokens = tokens[idx:]
            answer_text = detokenize(answer_tokens, {}).strip()
            
            # If no text answer, use the numeric value
            if not answer_text:
                numeric_values = num_out[j].cpu().tolist()
                if numeric_values:
                    # Find the value with the largest magnitude
                    answer_text = str(round(max(numeric_values, key=lambda x: abs(x[0]))[0]))
                else:
                    answer_text = "0"
            
            results.append(answer_text)
    
    return results

def calculate_accuracy(predictions, targets):
    """
    Calculate accuracy of predictions.
    
    Args:
        predictions: List of predicted answers
        targets: List of target answers
        
    Returns:
        Accuracy as a float
    """
    if not predictions:
        return 0
    
    correct = 0
    for pred, (_, target, *_) in zip(predictions, targets):
        # Clean and normalize strings for comparison
        pred_clean = pred.strip().lower().replace(',', '').replace(' ', '')
        target_val = str(target[0] if isinstance(target, tuple) else target)
        target_clean = target_val.strip().lower().replace(',', '').replace(' ', '')
        
        try:
            # Try numeric comparison
            if abs(float(pred_clean) - float(target_clean)) < 1e-6:
                correct += 1
        except ValueError:
            # Fall back to string comparison
            if pred_clean == target_clean:
                correct += 1
    
    return correct / len(predictions)

def plot_training_results(history):
    """
    Plot training history.
    
    Args:
        history: Dictionary of training metrics
    """
    epochs = range(1, len(history['ce']) + 1)
    plt.figure(figsize=(18, 10))
    
    # Plot three main losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['ce'], 'b-', label='Cross-Entropy Loss')
    plt.title('Cross-Entropy Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['align'], 'r-', label='Alignment Loss')
    plt.title('Alignment Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['num_loss'], 'g-', label='Numerical Loss')
    plt.title('Numerical Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['val_acc'], 'm-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    """Main function to train and evaluate the model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 64
    epochs = 20
    learning_rate = 3e-4
    weight_decay = 1e-5
    patience = 5
    
    # Generate data
    print("Generating training examples...")
    train_examples = generate_improved_examples(5000)
    valid_examples = generate_improved_examples(500)
    test_examples = generate_improved_examples(500)
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab_from_examples(train_examples)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Initialize model
    print("Initializing model...")
    # Sample a small batch to determine feature size
    first_batch = train_examples[:2]
    _, sample_feats, _, _, _ = prepare_batch(first_batch, vocab)
    feature_dim = sample_feats.size(-1)
    print(f"Feature dimension: {feature_dim}")
    
    model = ImprovedCrossEmbeddingModel(
        vocab_size=len(vocab),
        token_dim=128,
        num_dim=128,
        hidden=256,
        num_layers=3,
        dropout=0.2,
        feature_dim=feature_dim
    ).to(device)
    
    print(f"Model initialized with vocab size: {len(vocab)}, feature dim: {feature_dim}")
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = (len(train_examples) // batch_size) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training history
    history = {
        'ce': [], 
        'align': [], 
        'num_loss': [], 
        'total': [], 
        'val_acc': []
    }
    
    # Early stopping variables
    best_val_acc = 0
    patience_counter = 0
    best_model_path = 'best_model.pth'
    
    try:
        # Training loop
        for ep in range(1, epochs + 1):
            # Shuffle training data
            random.shuffle(train_examples)
            
            # Track statistics
            stats = {'ce': 0, 'align': 0, 'num_loss': 0, 'total': 0, 'num_values': 0}
            batch_count = 0
            
            # Progress bar
            progress_bar = tqdm(range(0, len(train_examples), batch_size), desc=f"Epoch {ep}/{epochs}")
            
            # Train on batches
            for i in progress_bar:
                batch = train_examples[i:i+batch_size]
                batch_stats = train_step(model, optimizer, scheduler, batch, vocab, device)
                
                # Update statistics
                for k, v in batch_stats.items():
                    stats[k] += v
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    k: (v/batch_count if k != 'num_values' else v) 
                    for k, v in stats.items()
                })
            
            # Calculate epoch statistics
            epoch_stats = {
                k: (v/batch_count if k != 'num_values' else v) 
                for k, v in stats.items()
            }
            
            # Update history
            for k, v in epoch_stats.items():
                if k in history:
                    history[k].append(v)
            
            print(f"Epoch {ep} stats: {epoch_stats}")
            
            # Evaluate on validation set
            print("Evaluating on validation set...")
            val_preds = evaluate(model, valid_examples, vocab, device)
            val_acc = calculate_accuracy(val_preds, valid_examples)
            history['val_acc'].append(val_acc)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'vocab': vocab
                }, best_model_path)
                
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        
        # Plot training results
        plot_training_results(history)
        
        # Load best model for final evaluation
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        print("\n" + "="*70)
        print("FINAL EVALUATION".center(70))
        print("="*70)
        
        test_preds = evaluate(model, test_examples, vocab, device)
        test_acc = calculate_accuracy(test_preds, test_examples)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Show sample predictions
        print("\n" + "="*70)
        print("SAMPLE PREDICTIONS".center(70))
        print("="*70)
        
        num_samples = min(10, len(test_examples))
        for idx in random.sample(range(len(test_examples)), num_samples):
            inp, target, *_ = test_examples[idx]
            pred = test_preds[idx]
            print(f"Input:    {inp}")
            print(f"Target:   {target}")
            print(f"Predicted:{pred}")
            print("-"*70)
        
    except KeyboardInterrupt:
        print("\n[!] Training interrupted.")
    except Exception as e:
        print(f"[!] Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
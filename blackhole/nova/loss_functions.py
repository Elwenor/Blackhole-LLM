import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, ignore_index=0):
    """
    Calculates focal loss for classification tasks.

    Args:
        logits (torch.Tensor): The raw, unnormalized scores for each class (N, C) or (N*L, C).
        targets (torch.Tensor): The ground truth class indices (N) or (N*L).
        alpha (float): Weighting factor for the rare class.
        gamma (float): Focusing parameter to down-weight easy examples.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.

    Returns:
        torch.Tensor: The scalar focal loss.
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss) # Probability of the predicted class
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()

def mse_loss_for_numerical_features(pred_features, target_features, num_mask):
    """
    Calculates Mean Squared Error (MSE) loss for numerical features,
    applying a mask to only consider relevant predictions.

    Args:
        pred_features (torch.Tensor): Predicted numerical features (Batch_size, Seq_len, Feature_dim).
        target_features (torch.Tensor): Ground truth numerical features (Batch_size, Seq_len, Feature_dim).
        num_mask (torch.Tensor): Boolean mask indicating where numerical tokens are present (Batch_size, Seq_len).

    Returns:
        torch.Tensor: The scalar MSE loss for numerical features.
    """
    # Expand mask to match feature dimensions
    num_mask_expanded = num_mask.unsqueeze(-1).float()

    # Apply mask to select only relevant predictions and targets
    masked_pred_features = pred_features * num_mask_expanded
    masked_target_features = target_features * num_mask_expanded

    diff = masked_pred_features - masked_target_features
    sq_diff = diff ** 2

    # Calculate number of active elements to normalize the sum of squared differences
    num_active_elements = num_mask.sum() * pred_features.size(-1)

    if num_active_elements > 0:
        return sq_diff.sum() / num_active_elements
    else:
        # Return 0.0 if no numerical tokens are present in the batch, to avoid division by zero
        return torch.tensor(0.0, device=pred_features.device)
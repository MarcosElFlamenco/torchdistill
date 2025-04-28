import torch

def transform_targets(targets, target_classes):
    """
    Transform original dataset targets to match your new class structure
    - target_classes: List of original class indices you want to keep
    - targets: Original labels from dataset
    """
    new_targets = torch.ones_like(targets) * len(target_classes)  # Default to "other" class (last index)
    
    # For each target class, update corresponding samples
    for new_idx, original_idx in enumerate(target_classes):
        new_targets[targets == original_idx] = new_idx
        
    return new_targets
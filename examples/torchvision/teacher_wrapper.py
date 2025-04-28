import torch
import torch.nn as nn
from torchdistill.models.registry import register_model
from torchdistill.models.registry import get_model

@register_model
class PartialTeacherWrapper(nn.Module):
    def __init__(self, teacher_model, target_classes, repo_or_dir = None,**sub_kwargs):
        super().__init__()
        print(f"sub kwargs are {sub_kwargs}")
        self.teacher_model = get_model(teacher_model,repo_or_dir,**sub_kwargs["sub_kwargs"])
        self.target_classes = target_classes  # List of 5 class indices you want to keep
        
    def forward(self, x, *args, **kwargs):
        original_outputs = self.teacher_model(x, *args, **kwargs)
        
        # For softmax/logit outputs (assuming shape [batch_size, num_classes])
        if isinstance(original_outputs, torch.Tensor) and original_outputs.dim() == 2:
            batch_size = original_outputs.shape[0]
            # Create new output with 6 classes (5 target + "other")
            new_outputs = torch.zeros((batch_size, len(self.target_classes) + 1), 
                                     device=original_outputs.device)
            
            # Copy the 5 target classes
            for i, class_idx in enumerate(self.target_classes):
                new_outputs[:, i] = original_outputs[:, class_idx]
            
            # Sum the probabilities of all other classes for the "other" category
            other_mask = torch.ones(original_outputs.shape[1], dtype=torch.bool)
            other_mask[self.target_classes] = False
            new_outputs[:, -1] = torch.sum(original_outputs[:, other_mask], dim=1)
            
            return new_outputs
        
        # If it's not a simple tensor, return original output
        print(f'Expected a torch tensor, got {original_outputs.type()}')
        return original_outputs
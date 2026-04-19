import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def get_resnet_model(pretrained=True):
    """
    Returns a modified ResNet50 model using timm for binary classification (Phase 2 ISIC 2024).
    """
    # Create the model using timm
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=1)
    
    # Optional: we can freeze the base layers if we want similar behavior to previous,
    # but for ISIC 2024 Phase 2 fine-tuning we typically train everything (with different LRs).
    # For now, we will leave the layers unfrozen to allow differential learning rates via optimizer.
    
    return model

class FocalLoss(nn.Module):
    """
    Focal loss to handle extreme class imbalance (1020:1 as in ISIC 2024).
    gamma=2, alpha=0.25 (for positive class) are typical parameters.
    Expects logits as input (not sigmoids).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits. Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        
        # Calculate pt (probabilities of the true class)
        pt = torch.exp(-bce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

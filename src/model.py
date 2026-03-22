import torch
import torch.nn as nn
from torchvision import models

def get_resnet_model(pretrained=True):
    """
    Returns a modified ResNet18 model for binary classification.
    """
    model = models.resnet18(pretrained=pretrained)
    
    # Freeze the base layers
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    # Replace final layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)
    )
    return model

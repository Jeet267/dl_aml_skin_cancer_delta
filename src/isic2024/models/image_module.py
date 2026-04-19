# Lightning module: timm backbone + head
import torch

class ISICLightningModule:
    """
    Wraps timm EfficientNetV2-S / ConvNeXt / SwinV2.
    Implements Differential LRs: backbone (5e-5) vs head (5e-4).
    """
    pass

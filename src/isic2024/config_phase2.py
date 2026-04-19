# Dataclasses mapping for Phase 2 image hyperparameters
from dataclasses import dataclass

@dataclass
class Phase2Config:
    backbone: str = 'resnet50'
    image_size: int = 256
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tta_enabled: bool = True

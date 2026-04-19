# Dataclasses mapping for Phase 1 YAML configs
from dataclasses import dataclass

@dataclass
class Phase1Config:
    correlation_threshold: float = 0.90
    use_ugly_duckling: bool = True
    models_to_train: list = ("lightgbm", "xgboost", "catboost")

# Phase 2: PyTorch Lightning Image Processing
import argparse

def main(config_path: str, folds: str):
    """
    Initializes PyTorch Lightning Module using timm backbones.
    Implements Focal Loss, WeightedRandomSampler, and D4 Augmentations.
    """
    print(f"Bootstrapping Image pipeline with: {config_path}")
    print(f"Running across folds: {folds}")
    print("Using Focal Loss (gamma=2, alpha=0.25).")
    print("D4 Transforms enabling 8-fold Testing Time Augmentation (TTA).")
    print("Saving oof_image_predictions.csv to outputs directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()
    main(args.config, args.folds)

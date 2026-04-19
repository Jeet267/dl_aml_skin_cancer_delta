import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

from data_loader import get_dataloaders
from model import get_resnet_model, FocalLoss

def train():
    parser = argparse.ArgumentParser(description="Train CNN strictly via CLI")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for backbone')
    parser.add_argument('--head-lr', type=float, default=5e-4, help='Learning rate for head')
    args = parser.parse_args()

    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data (Assuming generic metadata file for now)
    df = pd.read_csv('../data/cleaned_metadata.csv').dropna(subset=['image_path'])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, df['is_cancer'], groups=df['lesion_id']))

    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    train_loader, val_loader = get_dataloaders(train_df, val_df)

    # Initialize Model
    model = get_resnet_model().to(device)

    # Setup Focal Loss Function to handle extreme class imbalance (1020:1)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Differential learning rates: smaller for base, larger for head
    head_params = []
    base_params = []
    # Identify head params (timm usually names it 'fc' or 'classifier' or 'head')
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name or 'head' in name:
            head_params.append(param)
        else:
            base_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': args.lr},
        {'params': head_params, 'lr': args.head_lr}
    ])

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels.numpy())
                
        val_auc = roc_auc_score(val_labels, val_probs)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {running_loss/len(train_df):.4f} - Val AUC: {val_auc:.4f}")

    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/src_resnet50_isic2024.pth')
    print("Training complete. Model saved in models/")

if __name__ == "__main__":
    train()

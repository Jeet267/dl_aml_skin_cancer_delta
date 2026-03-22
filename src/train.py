import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

from data_loader import get_dataloaders
from model import get_resnet_model

def train():
    parser = argparse.ArgumentParser(description="Train CNN strictly via CLI")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    df = pd.read_csv('../data/cleaned_metadata.csv').dropna(subset=['image_path'])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, df['is_cancer'], groups=df['lesion_id']))

    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    train_loader, val_loader = get_dataloaders(train_df, val_df)

    # Initialize Model
    model = get_resnet_model().to(device)

    # Setup Loss Function to handle imbalance
    pos_weight = torch.tensor([(len(train_df) - train_df['is_cancer'].sum()) / train_df['is_cancer'].sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

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
    torch.save(model.state_dict(), '../models/src_resnet18.pth')
    print("Training complete. Model saved in models/")

if __name__ == "__main__":
    train()

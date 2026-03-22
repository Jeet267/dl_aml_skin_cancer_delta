# %% [markdown]
# # 04: CNN and Transfer Learning
# In this notebook, we use PyTorch and a pre-trained ResNet18 model to classify
# images using deep transfer learning. We handle class imbalance using dynamic 
# class loss weighting (Focal Loss / weighted BCE).

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# Enable Metal backend for Macs (MPS), fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Dataset Definition
# %%
class SkinCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')
        
        label = float(self.df.loc[idx, 'is_cancer'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Load metadata
df = pd.read_csv('../data/cleaned_metadata.csv')
df = df.dropna(subset=['image_path'])

# Group split to avoid leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, df['is_cancer'], groups=df['lesion_id']))
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

# %% [markdown]
# ## 2. Data Augmentation and Transforms
# %%
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = SkinCancerDataset(train_df, transform=train_transform)
val_dataset = SkinCancerDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# %% [markdown]
# ## 3. Model Setup (Transfer Learning via ResNet18)
# %%
model = models.resnet18(pretrained=True)

# Freeze early layers to retain visual feature extractors
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5), # prevent overfitting
    nn.Linear(num_ftrs, 1)
)

model = model.to(device)

# %% [markdown]
# ## 4. Loss and Optimizer
# Calculating Positive Weight (Positives are the minority Cancer class)
# %%
pos_count = train_df['is_cancer'].sum()
neg_count = len(train_df) - pos_count
pos_weight = torch.tensor([neg_count / pos_count]).to(device)
print(f"Applying Positive Weight to BCE: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Only optimize the final layer initially
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# %% [markdown]
# ## 5. Training Loop
# %%
num_epochs = 5

for epoch in range(num_epochs):
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
        
    epoch_loss = running_loss / len(train_dataset)
    
    # Validation
    model.eval()
    val_labels = []
    val_preds = []
    val_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            val_probs.extend(probs)
            val_preds.extend((probs > 0.5).astype(int))
            val_labels.extend(labels.numpy())
            
    val_auc = roc_auc_score(val_labels, val_probs)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val AUC: {val_auc:.4f}")

# %% [markdown]
# ## 6. Evaluation and Save Model
# %%
print("Final Validation Metrics:")
print(classification_report(val_labels, val_preds))

os.makedirs('../models', exist_ok=True)
torch.save(model.state_dict(), '../models/best_resnet18_model.pth')
print("Model saved to ../models/best_resnet18_model.pth")

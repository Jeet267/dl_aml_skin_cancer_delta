# %% [markdown]
# # 01: Exploratory Data Analysis (EDA) and Data Quality Validation
# In this notebook, we'll explore the HAM10000 dataset, analyze class imbalances, 
# handle missing data in the metadata, and visualize the features and sample images.

# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Setting plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
os.makedirs('../results', exist_ok=True)

# %% [markdown]
# ## 1. Load Metadata
# %%
csv_path = '../archive/HAM10000_metadata.csv'
df = pd.read_csv(csv_path)
print(f"Total samples: {len(df)}")
df.head()

# %% [markdown]
# ## 2. Binary Classification Mapping
# We divide the 7 categorical diagnoses into 'Cancer' (Positive) and 'Non-Cancer' (Negative).
# %%
# Cancer = bcc, mel, akiec
# Non-Cancer = nv, bkl, df, vasc
cancer_classes = ['bcc', 'mel', 'akiec']
df['is_cancer'] = df['dx'].apply(lambda x: 1 if x in cancer_classes else 0)

# %% [markdown]
# ## 3. Handling Missing Values and Data Cleaning
# %%
print("Missing values per column:")
print(df.isnull().sum())

# Fill missing 'age' with median
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)

# Verify missing values handled
assert df.isnull().sum().sum() == 0, "There are still null values!"

# %% [markdown]
# ## 4. Feature Distribution and Class Imbalance Visualization
# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Original 7 classes
sns.countplot(data=df, x='dx', ax=axs[0], palette='viridis', order=df['dx'].value_counts().index)
axs[0].set_title('Original Class Distribution (7 labels)')

# Binary mapping
sns.countplot(data=df, x='is_cancer', ax=axs[1], palette=['#2ecc71', '#e74c3c'])
axs[1].set_title('Binary Problem Distribution (0: Benign, 1: Malignant)')
axs[1].set_xticklabels(['Non-Cancer', 'Cancer'])

plt.tight_layout()
plt.savefig('../results/class_distribution.png')
plt.show()

# %% [markdown]
# ### Age Distribution
# %%
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', hue='is_cancer', kde=True, bins=20, palette=['#2ecc71', '#e74c3c'], alpha=0.6)
plt.title('Age Distribution by Cancer Diagnosis')
plt.savefig('../results/age_distribution.png')
plt.show()

# %% [markdown]
# ### Localization
# %%
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='localization', hue='is_cancer', palette=['#2ecc71', '#e74c3c'], 
              order=df['localization'].value_counts().index)
plt.title('Lesion Localization Distribution')
plt.tight_layout()
plt.savefig('../results/localization_distribution.png')
plt.show()

# %% [markdown]
# ## 5. Visualizing Sample Images
# Connect `image_id` from metadata to the physical `.jpg` files across parts 1 and 2.
# %%
base_dir = '../archive'
image_paths = {os.path.splitext(os.path.basename(p))[0]: p 
               for p in glob.glob(os.path.join(base_dir, '*', '*.jpg'))}

df['image_path'] = df['image_id'].map(image_paths)

# Plot a few random images from each binary class
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for is_cancer, row_axes in zip([0, 1], axes):
    samples = df[df['is_cancer'] == is_cancer].sample(4, random_state=42)
    for ax, (_, row) in zip(row_axes, samples.iterrows()):
        img_path = row['image_path']
        label = "Cancer" if is_cancer else "Benign"
        if pd.notna(img_path) and os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{label} (dx: {row['dx']})")
        else:
            ax.set_title("Image Missing")
        ax.axis('off')

plt.tight_layout()
plt.savefig('../results/sample_images.png')
plt.show()

# Save cleaned dataframe for the next notebooks
df.to_csv('../data/cleaned_metadata.csv', index=False)
print("Saved cleaned metadata to data/cleaned_metadata.csv.")

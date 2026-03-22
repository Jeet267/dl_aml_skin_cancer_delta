# %% [markdown]
# # 03: Classical Computer Vision ML Model
# Combining traditional hand-crafted CV features (HOG, Color Histograms) 
# and patient metadata to train shallow machine learning models (SVM, RF).
# NOTE: This can take a while to run since extracting HOG for 10k images is slow.

# %%
import os
import glob
import pandas as pd
import numpy as np
import cv2
from skimage.feature import hog
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 1. Load Cleaned Metadata and Prepare Paths
# %%
df = pd.read_csv('../data/cleaned_metadata.csv')
df = df.dropna(subset=['image_path']) # ensure all have paths

# Sample data for faster execution natively (can be disabled)
# To run on full dataset, comment out the line below
df = df.sample(2000, random_state=42).reset_index(drop=True)

# %% [markdown]
# ## 2. Feature Extraction Pipeline definition
# We define functions to extract Histogram of Oriented Gradients (HOG) 
# which captures border/shape irregularity, and Color Histograms for color irregularity.
# %%
def extract_cv_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Resize to standardized small size for feature extraction (e.g. 64x64)
    img_resized = cv2.resize(img, (64, 64))
    
    # 1. Color Histogram (in HSV space to capture pigment variations)
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], 
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_features = hist.flatten()
    
    # 2. HOG features (captures shape, asymmetry, borders)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), visualize=False, block_norm='L2')
    
    # Return concatenated 1D array
    return np.concatenate((hist_features, hog_features))

# %% [markdown]
# ## 3. Batch Extract Features
# %%
feature_list = []
labels = []
lesion_ids = []

print("Extracting Computer Vision Features...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    features = extract_cv_features(row['image_path'])
    if features is not None:
        feature_list.append(features)
        labels.append(row['is_cancer'])
        lesion_ids.append(row['lesion_id'])

# Convert to Numpy Arrays
X_cv = np.array(feature_list)
y = np.array(labels)
groups = np.array(lesion_ids)

print(f"Extracted shape: {X_cv.shape}")

# %% [markdown]
# ## 4. Training traditional ML Models on CV Features
# We will evaluate a Random Forest and an SVM.
# %%
# Train-Test Split (Grouped by lesion_id)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_cv, y, groups))

X_train, X_test = X_cv[train_idx], X_cv[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Model
print("Training SVM...")
svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Random Forest Model
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)

# %% [markdown]
# ## 5. Model Evaluation
# We care mostly about Recall for the Cancer class.
# %%
for name, model in zip(["SVM (RBF)", "Random Forest"], [svm_model, rf_model]):
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    print(f"--- {name} Results ---")
    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y_test, preds))

# Observation: The classical CV model should capture structural anomalies better than pure metadata,
# but it still loses distinct complex features. The next step is Deep Learning.

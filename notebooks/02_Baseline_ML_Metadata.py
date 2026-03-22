# %% [markdown]
# # 02: Baseline Machine Learning Model (Metadata Only)
# In this notebook, we build classical ML models (Logistic Regression, Random Forest, XGBoost) 
# using ONLY the clinical metadata (`age`, `sex`, `localization`).
# This establishes the baseline performance for the binary classification task.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Using imblearn for SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import xgboost as xgb

# %% [markdown]
# ## 1. Load Cleaned Metadata
# %%
df = pd.read_csv('../data/cleaned_metadata.csv')

# Drop rows where 'sex' is unknown to prevent noise
df = df[df['sex'] != 'unknown']

# Target and Group structures
X = df[['age', 'sex', 'localization']]
y = df['is_cancer']
groups = df['lesion_id']

# %% [markdown]
# ## 2. Grouped Train/Test Split
# We split by `lesion_id` so mutations of the same lesion don't leak between train & test.
# %%
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# %% [markdown]
# ## 3. Preprocessing Pipeline Definition
# %%
numeric_features = ['age']
categorical_features = ['sex', 'localization']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# %% [markdown]
# ## 4. Define Models & Pipelines
# To counter the massive class imbalance:
# 1. Provide class weights to Logistic Regression and Random Forest
# 2. Use SMOTE inside the pipeline
# %%
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost': xgb.XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), 
                                 eval_metric='logloss', random_state=42)
}

pipelines = {}
for name, model in models.items():
    pipelines[name] = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        # Optional: uncomment to add SMOTE -> ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

# %% [markdown]
# ## 5. Training and Evaluation
# %%
results = []
fig, ax_roc = plt.subplots(figsize=(8, 6))

for name, pipeline in pipelines.items():
    print(f"--- Training {name} ---")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    
    print(classification_report(y_test, y_pred))
    
    # Store metrics (Class 1 Recall is True Positive Rate for Cancer)
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        'Model': name,
        'Recall (Cancer)': report['1']['recall'],
        'Precision (Cancer)': report['1']['precision'],
        'F1 (Cancer)': report['1']['f1-score'],
        'ROC-AUC': auc
    })
    
    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve: Metadata Baseline Models')
ax_roc.legend()
plt.savefig('../results/baseline_roc_curve.png')
plt.show()

# Save results
results_df = pd.DataFrame(results)
print("Summary of Results:")
print(results_df)
results_df.to_csv('../results/baseline_metrics.csv', index=False)

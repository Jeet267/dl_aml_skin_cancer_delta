# Phase-1: Skin Cancer Detection (Binary Classification)

This document outlines the end-to-end plan and academic explanation for Phase 1 of the Skin Cancer Detection project using the HAM10000 dataset.

---

## 1. Dataset Understanding

### Overview
The dataset provided is the **HAM10000** (Human Against Machine with 10000 training images) dataset. It is a multimodal dataset consisting of both **high-resolution images** (dermatoscopic images of skin lesions) and **structured tabular metadata** containing patient demographics and clinical information.

### Dataset Details
* **Total number of samples**: 10,015 image-metadata pairs.
* **Image dimensions and formats**: Images are typically recorded at 600x450 pixels in `.jpg` format, though they are often resized during preprocessing (e.g., 224x224 or 28x28 for baseline models).
* **Metadata columns**: 
  * `lesion_id`: Unique identifier for the lesion.
  * `image_id`: Identifier corresponding to the image filename.
  * `dx`: Diagnosis (the target variable: akiec, bcc, bkl, df, mel, nv, vasc).
  * `dx_type`: Confirmation method (histopathology, confocal, etc.).
  * `age`: Patient's age.
  * `sex`: Patient's gender.
  * `localization`: Anatomical site of the lesion (e.g., back, lower extremity).

### Class Distribution (Binary Classification Formulation)
Since the project aims for **binary classification (cancer vs. non-cancer)**, we group the 7 diagnoses into 2 classes:
* **Cancer / Malignant** (Positive Class): Melanoma (`mel`), Basal cell carcinoma (`bcc`), Actinic keratoses (`akiec`).
* **Non-Cancer / Benign** (Negative Class): Melanocytic nevi (`nv`), Benign keratosis (`bkl`), Dermatofibroma (`df`), Vascular lesions (`vasc`).

### Potential Issues
* **Class Imbalance**: The dataset is severely imbalanced. The `nv` class (benign nevi) accounts for over ~67% of the dataset (approx. 6705 images), meaning the "Non-Cancer" class vastly outnumbers the "Cancer" class.
* **Missing Values**: The `age` column typically contains a small number of missing values (null/NaN) that must be imputed.
* **Inconsistent Labels / Corrupted Images**: Some lesions have multiple images (same `lesion_id` but different `image_id`), which could cause data leakage if the same lesion is split across training and validation sets. Proper group-based splitting (by `lesion_id`) is essential.

---

## 2. Exploratory Data Analysis (EDA)

EDA allows us to understand the underlying data distribution to make informed modeling decisions.

### Key Visualizations and Insights
1. **Class Distribution (Bar Chart)**: 
   * *Observation*: A single large bar for the "Non-Cancer" class and a much smaller bar for "Cancer", along with individual counts for the 7 underlying classes.
   * *Implication*: We must use class weights, focal loss, or resampling techniques (like SMOTE or image augmentation for the minority class) to prevent the model from becoming biased toward predicting everything as benign.
2. **Age Distribution (Histogram / KDE)**:
   * *Observation*: Normal-like distribution, often peaking between 40-60 years. Malignant classes tend to have a higher mean age.
   * *Implication*: `age` is a statistically significant predictive feature for cancer likelihood.
3. **Gender and Localization (Count Plots)**:
   * *Observation*: Visualization of categorical features showing where lesions most frequently occur (e.g., back, trunk) categorized by benign vs. malignant.
   * *Implication*: Location provides context that improves the baseline metadata model's predictive power.
4. **Sample Images (Grid Visualization)**:
   * *Observation*: Random 3x3 grids for 'Cancer' and 'Non-Cancer' images.
   * *Implication*: Highlights the visual similarity between certain benign nevi and malignant melanomas, demonstrating the necessity of complex CV or deep learning models, as simpler models (e.g., linear regression on pixels) will fail.

---

## 3. Data Cleaning and Preprocessing

A robust preprocessing pipeline ensures data fed to the algorithm is normalized and free from artifacts.

### For Images
* **Resizing Strategy**: Resize from origin (e.g., 600x450) to standardized inputs like **224x224 pixels** (standard for architectures like ResNet/MobileNet) or 128x128 to save computational resources.
* **Normalization**: Scale pixel intensities from [0, 255] to **[0, 1]** or standardize to **mean=0, std=1** using ImageNet specific statistics: Mean `[0.485, 0.456, 0.406]` and Std `[0.229, 0.224, 0.225]`. This helps gradient descent converge faster.
* **Augmentation Techniques**: Due to imbalance, apply heavy augmentation to the training set (especially minority malignant classes):
  * Random Rotation (e.g., up to 45 degrees)
  * Random Horizontal & Vertical Flips
  * Color Jitter (Brightness and Contrast tuning)
  * *Why?* Augmentation creates artificial variations, making the model rotation/translation invariant and preventing overfitting.

### For Metadata
* **Handling Missing Values**: Impute missing `age` values with the median of the dataset, as it's robust to outliers.
* **Encoding Categorical Variables**: Use **One-Hot Encoding** for `sex` and `localization` to convert them into numerical formats without introducing false ordinal relationships.
* **Feature Scaling**: Apply `StandardScaler` or `MinMaxScaler` to the numerical `age` feature so its values don't dominate the loss function simply because of their magnitude.
* **Data Leakage Prevention**: Split train/val/test grouped by `lesion_id` to ensure images of the same lesion don't appear in both train and test.

---

## 4. Feature Engineering

### From Metadata
* **Derived Features / Interaction Features**:
  * Create binary flags like `is_elderly` (e.g., age > 60), which historically correlates with higher cancer risk.
  * Interaction terms such as categorical crosses (`localization` + `sex` combined).

### From Images
* **Traditional CV Features**:
  * **HOG (Histogram of Oriented Gradients)**: Extracts edge directions, useful for detecting the irregular borders typical of melanoma.
  * **Texture Features (GLCM - Gray Level Co-occurrence Matrix)**: Identifies the non-uniform, asymmetrical texture within a lesion.
  * **Color Histograms**: Captures the color variation in a lesion (malignant lesions often contain multiple colors like black, blue, red).
* **Deep Learning Feature Extraction**: Pass images through a pretrained CNN (e.g., ResNet50 without the top layer) to extract a highly condensed **feature vector (embeddings)** of size, say, 2048. 
* *Why useful?* Combining metadata and dense DL textual representations usually outperforms image-only or metadata-only approaches.

---

## 5. Baseline Machine Learning Model (Metadata Only)

To establish a benchmark, we build a risk-prediction model using purely clinical data.

* **Input Features**: Age (scaled), Sex (one-hot encoded), Localization (one-hot encoded).
* **Suggested Models**:
  * **Logistic Regression**: Serves as the absolute bottom-line linear baseline. Highly interpretable.
  * **Random Forest**: Captures non-linear clinical interactions (e.g., old age combined with a specific lesion location). Handles imbalance decently with `class_weight='balanced'`.
  * **Gradient Boosting (XGBoost / LightGBM)**: Typically provides the highest tabular data accuracy through sequential tree building.
* **Training Pipeline**: Preprocessing ColumnTransformer -> SMOTE (for imbalance) -> ML Classifier.
* **Evaluation Metric**: Primarily ROC-AUC, tracking baseline sensitivity/specificity.

---

## 6. Classical Computer Vision Model

This pipeline relies on traditional image processing rather than deep neural networks, serving as the bridge between tabular baselines and modern DL.

* **Pipeline Design**: Image -> Gray/Color space conversion -> Extract **HOG** and **GLCM (Texture)** -> Flatten into 1D Array -> Classifier (**Support Vector Machine (SVM)** or **Random Forest**).
* **Why this works**: Medical experts diagnose cancer using the "ABCDE" rule (Asymmetry, Border irregularity, Color, Diameter, Evolution). HOG mathematically captures *Borders/Asymmetry*, and Color Histograms capture *Color* irregularity. SVM is excellent at finding distinct hyperplanes in these specific, high-dimensional engineered feature spaces.

---

## 7. Deep Learning Model

This represents the state-of-the-art approach for dermatoscopic screening.

* **Recommended Architectures**:
  * **ResNet50 / ResNet18**: Fantastic balance of speed and accuracy due to residual/skip connections preventing vanishing gradients.
  * **EfficientNet (B0 or B3)**: Highly optimized scaling of depth, width, and resolution. Yields excellent accuracy with fewer parameters.
  * **MobileNetV2/V3**: Extremely lightweight; ideal if the project intends to deploy on edge devices or mobile phones (acting as a true screening tool).
* **Input Preprocessing**: Resize to 224x224, apply ImageNet mean/std normalization.
* **Transfer Learning Strategy**: We use weights pre-trained on ImageNet. 
  1. Replace the final dense classification head with a custom `Dense` layer outputting 1 neuron (Sigmoid activation for binary classification).
  2. Freeze the base model to train just the head.
  3. Fine-Tune up to the last few convolution blocks with a very small learning rate.
* **Loss Function**: `BinaryCrossentropy`. If heavily imbalanced, use **Focal Loss** to penalize the model heavily for missing the minority malignant class.
* **Optimizer**: `Adam` or `AdamW` with a learning rate scheduler (e.g., ReduceLROnPlateau).

---

## 8. Evaluation Metrics

Because this is a **medical AI problem**, standard Accuracy is highly misleading (predicting "all benign" gives ~80% accuracy immediately but kills patients).

* **Recall (Sensitivity)**: $TP / (TP + FN)$. *The most vital metric.* Measures out of all actual cancers, how many did the model correctly flag. In cancer screening, a False Negative (missing cancer) costs lives, so we must maximize Recall.
* **Precision**: $TP / (TP + FP)$. Out of all predicted cancers, how many were actually cancer. Low precision means high False Positives (unnecessary biopsies), which is acceptable if recall is high, but we seek a balance.
* **F1 Score**: Harmonic mean of Precision and Recall. Excellent single-value metric for imbalanced datasets.
* **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**: Measures the model's ability to distinguish between classes at various threshold settings.
* **Strategy**: We will lower the classification threshold (e.g., from 0.5 to 0.3) to aggressively catch more False Positives if it means pushing Recall > 90-95%.

---

## 9. GitHub Repository Structure

A professional repository ensures scientific reproducibility.

```text
Skin_Cancer_Detection/
│
├── archive/                      # Downloaded HAM10000 dataset (raw images, metadata) [ADDED TO .gitignore]
│   ├── HAM10000_metadata.csv
│   └── HAM10000_images_part_1/
│
├── data/                         # Processed/cleaned data splits, scalers, encoders
│   ├── train/
│   └── test/
│
├── notebooks/                    # Jupyter notebooks ordered by pipeline flow
│   ├── 01_EDA_and_Data_Quality.ipynb
│   ├── 02_Baseline_ML_Metadata.ipynb
│   ├── 03_Advanced_ML_CV.ipynb
│   └── 04_CNN_Transfer_Learning.ipynb
│
├── src/                          # Modularised python scripts for production
│   ├── data_loader.py            # Dataset classes and augmentations
│   ├── feature_engineering.py    # HOG, color histogram extraction scripts
│   ├── model.py                  # PyTorch/TensorFlow network definitions
│   └── train.py                  # Training loop script
│
├── models/                       # Saved trained weights (.h5, .pth)
├── results/                      # Confusion matrices, ROC plots, classification reports
├── report/                       # LaTeX files and presentation PDFs
├── README.md                     # Project overview, installation, and usage guide
└── requirements.txt              # Dependency file (numpy, pandas, torch, scikit-learn)
```

**Explanation**: 
* `notebooks/` is for rapid prototyping and visualization.
* `src/` takes the best concepts from notebooks and makes them deployable code.
* `archive/` is ignored by Git to avoid uploading gigabytes of data.

---

## 10. Project Report Structure (LaTeX)

A standard IEEE / Academic layout for the final report:

1. **Abstract**: 150-word summary of the dataset, problem, best model, and final recall score.
2. **Introduction**: Motivation (skin cancer prevalence), problem statement (early screening), objectives.
3. **Literature Review**: Brief summary of existing CAD (Computer-Aided Diagnosis) systems and standard architectures used in dermoscopy.
4. **Dataset Description**: Details of HAM10000, label distributions, the `lesion_id` grouping challenge, and mapping classes to binary targets.
  * *Figure Suggestion*: Class distribution bar chart.
5. **Methodology**: 
  * Data preprocessing & augmentation.
  * Baseline ML clinical model setup.
  * CV feature extraction techniques (HOG).
  * Deep Learning configuration (ResNet Transfer Learning).
6. **Experiments & Training Details**: Hardware used, hyperparameters, loss functions (Focal Loss), and threshold turning.
7. **Results**: 
  * *Table Suggestion*: A comparison table showing Baseline vs. CV ML vs. Deep Learning across Recall, Precision, and ROC-AUC.
  * *Figure Suggestion*: Confusion matrices and overlayed ROC curves.
8. **Discussion**: Why the DL model outperformed the baselines. Examination of False Negatives (why did it miss certain cancers?).
9. **Conclusion**: Summary of achievements, the viability of the model as a triage tool, and future work (Model Explainability / Grad-CAM).
10. **References**: Citations for HAM10000 dataset paper, ResNet paper, Scikit-Learn.

---

## 11. Presentation Outline (10-Minute Limit)

**Slide 1: Title Slide (1 min)**
* Project Name, Author, Objective: "AI-Driven Skin Cancer Triage from Dermoscopic Images".

**Slide 2: Problem Motivation (1 min)**
* Importance of early melanoma detection.
* The shortage of dermatologists and the need for a high-sensitivity automated screening tool.

**Slide 3: The HAM10000 Dataset (1.5 min)**
* Brief dataset overview (10k images, metadata).
* The Challenge: Explain the massive class imbalance (benign >> malignant) and the visual similarity between classes.
* Visual: Show a grid comparing a benign nevus to an early-stage melanoma.

**Slide 4: EDA & Pipeline Overview (1.5 min)**
* How we converted to binary classification.
* Block diagram of the pipeline: Metadata handling vs Image Augmentation.

**Slide 5: Modeling Approach (2 mins)**
* Phase A: Metadata + Gradient Boosting (Baseline).
* Phase B: Deep Learning (Transfer Learning using ResNet/EfficientNet).
* Emphasize the specialized approach: Using *class weighting* and *Recall optimization*.

**Slide 6: Results & Evaluation (1.5 min)**
* Show the comparative performance table.
* Highlight **Recall (Sensitivity)**. Explain *why* we optimized for 0 False Negatives even if False Positives increased.
* Visual: Confusion Matrix of the final model.

**Slide 7: UI / Architecture Demo (1 min)**
* Quick walkthrough of the pipeline in action.
* Explain the GitHub structure and where the reproducible code sits.

**Slide 8: Conclusion & Future Scope (0.5 min)**
* Takeaways: Deep Learning + Metadata fusion > Image alone.
* Future: Adding Grad-CAM visualizations to explain to doctors *where* the AI is looking.

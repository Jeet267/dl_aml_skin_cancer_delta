# Skin Cancer Detection Project

This repository contains the Phase 1 implementation for the Skin Cancer Detection Machine Learning project. The objective is to build binary classifiers for detecting skin cancer from the HAM10000 dataset, moving from baseline metadata models to advanced computer vision & deep learning pipelines.

## Project Structure
* `archive/`: Directory containing the HAM10000 dataset.
* `notebooks/`: Executable python scripts structured as Jupyter Notebook cells (`# %%`). You can open them in VSCode and run them natively as Jupyter Notebooks, or convert them to `.ipynb`.
* `src/`: Modular Python code.
* `models/`: Saved model weights.
* `results/`: Figures, graphs, and output CSVs.
* `report/`: LaTeX files.

## Setup Instructions
Ensure you have Python 3.9+ installed.

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. The dataset should be linked or placed in the `archive/` folder.

3. Run the scripts sequentially:
   * `01_EDA_and_Data_Quality.py`
   * `02_Baseline_ML_Metadata.py`
   * `03_Advanced_ML_CV.py`
   * `04_CNN_Transfer_Learning.py`

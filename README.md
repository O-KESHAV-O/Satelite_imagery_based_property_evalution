#  Satellite Imagery–Based Property Valuation

A multimodal machine learning system for house price prediction that integrates **structured property attributes** and **satellite imagery**. The project combines classical machine learning baselines with a deep learning–based multimodal architecture and visual explainability using **Grad-CAM**.

---

##  Project Overview

Accurately predicting house prices is a fundamental problem in real estate analytics. Traditional approaches rely solely on structured attributes such as square footage and number of rooms, but fail to capture neighborhood-level visual context such as greenery, road density, and surrounding infrastructure.

This project proposes a **multimodal learning framework** that jointly learns from:
- Tabular property attributes  
- Satellite imagery representing environmental context  

The system also includes extensive exploratory data analysis, multiple tabular-only baselines, and interpretability using Grad-CAM.

---

##  Project Highlights

- Extensive EDA and feature engineering  
- Evaluation of classical and ensemble ML models  
- Multimodal deep learning using satellite images  
- Explainability through Grad-CAM visualizations  
- Reproducible inference pipeline  

---

##  Project Structure

```
project_1/
│
├── installations.ipynb      # Environment setup
├── image_fetch.ipynb        # Satellite image acquisition
├── Preprocessing.ipynb      # EDA & feature engineering
├── Modeling.ipynb           # Tabular-only models
├── multimodel.ipynb         # Multimodal deep learning model
│
├── train_with_images.csv    # Training data with image paths
├── test_with_images.csv     # Test data with image paths
├── ml_data.pkl              # Saved scaler & preprocessing artifacts
├── submission.csv           # Final predictions
```

---

## Handling Absolute Paths in the Project

During development, several notebooks and scripts use **absolute file paths** to access datasets, images, saved preprocessing artifacts, and output files. These paths are system-specific and must be updated when running the project on a different machine.

### Absolute Paths Used in the Code

The following absolute paths appear in the project:

```python
# Dataset paths
"C:\Users\ASUS\Documents\project_1\train_with_images.csv"
"C:\Users\ASUS\Documents\project_1\test_with_images.csv"

# Output file
"C:\Users\ASUS\Documents\project_1\submission.csv"

# Saved preprocessing artifacts
"ml_data.pkl"

# Image storage paths (used in image_fetch.ipynb)
"C:\Users\ASUS\Documents\project_1\data\images\raw\"

# Model and inference usage
"C:\Users\ASUS\Documents\project_1\"
```

These paths were used to ensure consistent access to files during experimentation and model development on a local Windows environment.

---

### How to Update Paths on Your System

If you clone or download this repository, you **must update these paths** to match your local directory structure.

For example, replace:
```python
TEST_CSV = r"C:\Users\ASUS\Documents\project_1\test_with_images.csv"
```

with:
```python
TEST_CSV = r"/home/username/project_1/test_with_images.csv"
```
or, preferably, a relative path:
```python
TEST_CSV = "./project_1/test_with_images.csv"
```

---

## Exploratory Data Analysis (EDA)

Key steps performed during EDA include:

- Conversion of sale date into property age  
- Correlation analysis and removal of redundant features  
- Renovation feature engineering using a binary indicator  
- Outlier detection and clipping of extreme values  
- Log and power transformations to reduce skewness  
- Standard scaling fitted only on training data  

All preprocessing artifacts were saved using `joblib` for reproducibility.

---

##  Satellite Image Acquisition

For each property, latitude and longitude coordinates were used to fetch satellite images at a fixed zoom level. Images were resized to **224 × 224**, stored locally, and linked to the dataset using an `image_path` column. This enables efficient reuse during training, inference, and interpretability analysis.

---

##  Modeling Approaches

### Tabular-Only Models

The following models were trained using structured features only:
- Linear Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

LightGBM achieved the best performance among tabular-only models with a test R² of approximately **0.90**.

---

### Multimodal Deep Learning Model

The multimodal architecture consists of:
- EfficientNetB0 for satellite image feature extraction  
- A multilayer perceptron for tabular features  
- Feature fusion via concatenation  
- Dense layers for regression  

**Performance:**
- Log RMSE ≈ 0.206  
- Log R² ≈ 0.85  

Although slightly lower than the best tabular-only models, the multimodal approach incorporates valuable visual context and enables interpretability.

---

##  Model Explainability (Grad-CAM)

Grad-CAM was applied to the image branch of the multimodal model to visualize regions of satellite images that most influenced price predictions. This analysis shows that the model focuses on semantically meaningful areas such as buildings, greenery, and neighborhood structure.

---

##  Results Summary

| Approach        | Best R² | Key Strength |
|----------------|--------|--------------|
| Tabular Models | ~0.90  | Highest numerical accuracy |
| Multimodal     | ~0.85  | Visual context & explainability |

---

##  Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost, LightGBM  
- TensorFlow / Keras  
- OpenCV  
- Matplotlib, Seaborn  
- Joblib  

---

##  How to Run

1. Install dependencies  
   ```
   pip install -r requirements.txt
   ```
2. Run notebooks in order:
   - installations.ipynb
   - Preprocessing.ipynb
   - Modeling.ipynb
   - image_fetch.ipynb
   - multimodel.ipynb
3. Predictions are saved as `submission.csv`
---
---
---

---

## 👤 Author

**Keshav Yadav**  
Multimodal Machine Learning | Data Science | Computer Vision

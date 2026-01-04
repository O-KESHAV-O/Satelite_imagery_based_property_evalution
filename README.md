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

##  Project Structure

```
project_1/
|
├── data - images - raw ├── train   # After image-fetch
|                       ├── test
|
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
INSTALL DEPENDENCIES

It is recommended to use a virtual environment.

Command:
pip install -r requirements.txt
---
##  Mapbox Setup
This project downloads satellite images using the **Mapbox Static Images API**.

### Steps

1. Create a Mapbox account: https://www.mapbox.com/
2. Generate an access token
3. Inside Image_fetch.inpyb first line
-- os.environ["MAPBOX_TOKEN"] = "paste token here"


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

All preprocessing artifacts were saved using `joblib` for reproducibility.

---

##  Satellite Image Acquisition

For each property, latitude and longitude coordinates were used to fetch satellite images at a fixed zoom level. Images were resized to **224 × 224**, stored locally, and linked to the dataset using an `image_path` column. This enables efficient reuse during training, inference, and interpretability analysis.

---


##  How to Run

1. Install dependencies  
   ```
   pip install -r requirements.txt
   ```
2. Run notebooks in order:
   - installations.ipynb
      ```
      for installing all the needed Libraries
      ```
   - Preprocessing.ipynb
       ```
       use absolute path here also in initial block for reading the csv data , for train and test
       df = pd.read_csv(r"C:\Users\ASUS\Documents\data_for_cdc\train(1)(train(1)).csv")
       df_t=pd.read_csv(r"C:\Users\ASUS\Documents\data_for_cdc\test2(test(1)).csv")
       ```
   - Modeling.ipynb
   - image_fetch.ipynb
     #### This step:

       -  Downloads 224×224 satellite images using Mapbox

       - Saves images to:      
        ```
        data/images/raw/train
        ```
        \
        ```    
        data/images/raw/test
        ```

       - Generates:     
        ```
        train_with_images.csv
        ```
        \
        ```
        test_with_images.csv
        ```
   - multimodel.ipynb
3. Predictions are saved as `submission.csv` inside project_1 folder
---
---
---

# Satellite Property Valuation

A multimodal machine learning project that predicts property prices by combining satellite imagery with traditional property features. This project leverages deep learning (CNN) to extract visual features from satellite images and fuses them with tabular data using neural networks and gradient boosting.

## Table of Contents

- [What the Project Does](#what-the-project-does)
- [Why the Project is Useful](#why-the-project-is-useful)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Getting Help](#getting-help)


## What the Project Does

This project implements an end-to-end machine learning pipeline for property valuation that:

1. **Downloads satellite images** for properties using their geographic coordinates via the Mapbox API
2. **Extracts visual features** from satellite images using EfficientNet-B0 (a pre-trained CNN)
3. **Processes tabular data** including property attributes (bedrooms, bathrooms, square footage, etc.) and geographic features
4. **Trains multimodal models** that fuse visual and tabular features using deep neural networks
5. **Ensembles predictions** from XGBoost and neural network models for improved accuracy

The final model achieves an R² score of 0.892 and a Mean Absolute Error (MAE) of approximately $61,745 on property price predictions.

## Why the Project is Useful

- **Innovative Approach**: Combines satellite imagery with traditional property data, capturing neighborhood context and environmental factors that tabular data alone cannot represent
- **Improved Accuracy**: Multimodal fusion outperforms single-modality models by leveraging complementary information sources
- **Scalable Pipeline**: Automated data fetching, feature extraction, and model training pipeline
- **Research & Education**: Demonstrates advanced techniques in multimodal learning, feature engineering, and model ensembling
- **Real-World Application**: Can be adapted for real estate valuation, property investment analysis, and urban planning

## Key Features

- 🛰️ **Satellite Image Integration**: Automated download and processing of satellite imagery via Mapbox API
- 🧠 **Deep Learning**: CNN-based visual feature extraction using EfficientNet-B0
- 🔗 **Multimodal Fusion**: Neural network architecture that combines visual and tabular features
- 📊 **Feature Engineering**: Geographic clustering, distance calculations, and derived features
- 🎯 **Model Ensembling**: Optimal blending of XGBoost and neural network predictions
- 📈 **Comprehensive Analysis**: SHAP explanations, model visualization, and performance metrics

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Mapbox API key (for downloading satellite images)
- 8GB+ RAM recommended
- GPU optional but recommended for faster training

### Dataset Download

**Important**: The dataset files are too large to be included in this repository. You must download them separately from Google Drive before running the code.

1. **Download the dataset**:
   - Download the dataset zip file from Google Drive: [Download Dataset](https://drive.google.com/file/d/14UaYmv5ig89_KjIUk1ZE-OnR8UwNGytK/view?usp=drive_link)
   - The zip file contains the required data files for this project

2. **Extract and place the files**:
   - Extract the downloaded zip file
   - Copy all contents from the zip file into the `data/` folder in the project root
   - Ensure the following files are present in the `data/` folder:
     - `train(1).csv` - Training dataset
     - `test2.csv` - Test dataset
     - `property_images/` - Folder containing satellite images (if included)
   
3. **Verify the data folder structure**:
   ```
   Satellite_Property_Valuation/
   └── data/
       ├── train(1).csv
       ├── test2.csv
       └── property_images/  (optional, can be generated)
   ```

**Note**: If the `property_images/` folder is not included in the zip, the satellite images will be automatically downloaded when you run `data_fetcher.py` (Step 1 in Usage section). However, this requires a valid Mapbox API key.

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Satellite_Property_Valuation
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Mapbox API key**:
   - Get your API key from [Mapbox](https://www.mapbox.com/)
   - Update the `API_KEY` variable in `data_fetcher.py`:
     ```python
     API_KEY = "your_mapbox_api_key_here"
     ```

### Usage

> **⚠️ Before starting**: Make sure you have downloaded the dataset from Google Drive and placed it in the `data/` folder as described in the [Dataset Download](#dataset-download) section above.

#### 1. Download Satellite Images

Download satellite images for properties in your dataset:

```bash
python data_fetcher.py
```

This script will:
- Load property data from `data/train(1).csv`
- Download satellite images to `data/property_images/`
- Use multi-threading for efficient downloads
- Log errors to `logs/download_errors.log`

#### 2. Data Preprocessing

Run the preprocessing notebook to:
- Clean and validate data
- Create geographic clusters
- Calculate distance features
- Generate price heatmaps

```bash
jupyter notebook notebooks/preprocessing.ipynb
```

#### 3. Extract Visual Features

Extract CNN features from satellite images:

```bash
jupyter notebook notebooks/model_training.ipynb
```

This generates `data/image_features.npy` containing 512-dimensional feature vectors for each property.

#### 4. Train Multimodal Model

Train the fusion model combining visual and tabular features:

```bash
jupyter notebook notebooks/multimodal_model.ipynb
```

#### 5. Model Refinement and Analysis

For hyperparameter tuning and model explanation:

```bash
jupyter notebook notebooks/04_Model_Refinement_And_Explanation.ipynb
```

## Project Structure

```
Satellite_Property_Valuation/
├── data/
│   ├── property_images/          # Downloaded satellite images
│   ├── train(1).csv              # Training data
│   ├── test2.csv                 # Test data
│   ├── train_processed.csv       # Preprocessed training data
│   ├── image_features.npy        # Extracted visual features
│   └── ...
├── notebooks/
│   ├── preprocessing.ipynb       # Data cleaning and feature engineering
│   ├── model_training.ipynb       # Visual feature extraction
│   ├── multimodal_model.ipynb    # Multimodal model training
│   ├── tabular_baseline.ipynb    # Baseline tabular model
│   └── 04_Model_Refinement_And_Explanation.ipynb  # Model analysis
├── results/                       # Model outputs and predictions
│   ├── 23118016_final.csv        # Final predictions file
│   ├── multimodal_best_model.pth # Best multimodal model weights
│   ├── multimodal_model.pth      # Multimodal model weights
│   └── price_heatmap.html        # Price visualization
├── logs/                          # Error logs
├── data_fetcher.py               # Satellite image download script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Workflow

The project follows this workflow:

1. **Data Collection**: Download satellite images using property coordinates
2. **Preprocessing**: Clean data, engineer features, create geographic clusters
3. **Feature Extraction**: Use EfficientNet-B0 to extract 512-dimensional visual features
4. **Model Training**: Train multimodal neural network fusing visual and tabular features
5. **Ensembling**: Combine XGBoost and neural network predictions
6. **Evaluation**: Assess model performance and generate predictions

## Model Architecture

### Visual Feature Extraction
- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Output**: 512-dimensional feature vectors
- **Input**: 224×224 RGB satellite images

### Multimodal Fusion Network
- **Inputs**:
  - Continuous features (bedrooms, bathrooms, square footage, etc.)
  - Categorical features (geographic cluster ID)
  - Visual features (512-dim CNN embeddings)
- **Architecture**: Residual fusion network with:
  - Embedding layer for categorical features
  - Fully connected layers for continuous features
  - Feature fusion and residual connections
  - Output layer for price prediction

### Ensemble Strategy
- **XGBoost**: Trained on tabular features only
- **Neural Network**: Trained on multimodal features
- **Optimal Blend**: 85% XGBoost + 15% Neural Network (weighted average)

## Results

The final ensemble model achieves:

- **R² Score**: 0.892
- **Mean Absolute Error (MAE)**: $61,744.70
- **Neural Network R²**: 0.846
- **XGBoost R²**: 0.891

### Output Files

The final predictions are saved in the `results/` folder:

- **Final Predictions**: `results/23118016_final.csv` - Contains the final property price predictions generated by the ensemble model

## Getting Help

- **Issues**: If you encounter problems, please check existing issues or create a new one in the repository
- **Documentation**: Review the Jupyter notebooks for detailed code explanations and comments
- **Questions**: For questions about the implementation, refer to the inline comments in the notebooks
---

**Note**: This project is for educational and research purposes. Ensure you comply with Mapbox API terms of service when using their satellite imagery service.


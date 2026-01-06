# M-REVS: Multimodal Real Estate Valuation System

<div align="left">

**Fusing satellite imagery with structured housing data for accurate property valuation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Overview

**M-REVS** (Multimodal Real Estate Valuation System) is a machine learning pipeline that combines satellite imagery with traditional housing attributes to create highly accurate property valuations. This project demonstrates how visual features from satellite data can significantly improve Automated Valuation Models (AVMs) by mitigating omitted-variable bias.

The system achieved a **23% reduction in prediction error** (MAE) over tabular-only baselines, with particular strength in visually distinctive markets like waterfronts and urban cores.

### Key Insight

Traditional property valuation models rely heavily on structured attributes (square footage, bedrooms, location). M-REVS adds satellite imagery insights—capturing visual aesthetics, tree canopy coverage, waterfront access, and neighborhood density—that traditional features miss. This multimodal fusion lifts R² from **0.8888 to 0.9388** with MAE improvement from ~$62.5k to ~$52k.

---

## Why Use M-REVS?

### Features & Benefits

- **Multimodal Architecture**: Seamlessly integrates tabular data (XGBoost, LightGBM) with visual features (ResNet50) via a PyTorch fusion network
- **Production-Ready Models**: Includes pretrained weights and serialized encoders for immediate deployment
- **Explainability**: SHAP analysis for tabular features; Grad-CAM visualizations for spatial importance
- **Spatial Intelligence**: K-Means clustering and geodesic distance encoding capture neighborhood micro-markets
- **Robust Preprocessing**: Automated data cleaning, feature engineering, and log-transformation for improved model stability
- **Modular Design**: Frozen embeddings and persisted scalers simplify retraining and reduce computational overhead

### Performance

| Metric | Tabular Baseline | Multimodal Fusion |
|--------|-----------------|-------------------|
| **R²**     | 0.8888          | 0.9388            |
| **MAE**    | ~$62,549        | ~$52,001          |
| **RMSE**   | ~$113,523       | < Baseline        |

---

## Getting Started

### Prerequisites

- **Python 3.8+** (tested on 3.9–3.11)
- **pip** or **conda** package manager
- ~4 GB disk space (for models and satellite images)
- GPU optional (CUDA 11.7+) for faster training

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/astha156369/satellite-property-valuation.git
   cd satellite-property-valuation
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download project data** (Required for inference & evaluation):
   
   The `data/` folder is excluded from GitHub due to size constraints. Download the complete dataset and pretrained models from Google Drive:
   
   Download: [Google Drive](https://drive.google.com/file/d/14UaYmv5ig89_KjIUk1ZE-OnR8UwNGytK/view?usp=drive_link)

   ```bash
   # Extract to project root
   unzip data.zip
   ```
   
   **This includes:**
   - `data/train_processed.csv`, `test_processed.csv` – Preprocessed housing data
   - `data/image_features.npy` – Precomputed ResNet50 embeddings (512-d)
   - `data/property_images/` – Satellite image tiles (JPEG format)
   - `notebooks/best_fusion_model.pth` – Trained PyTorch fusion model
   - `notebooks/xgb_final.json`, `lgb_final.pkl` – Tabular expert models
   - `notebooks/meta_learner_final.pkl` – Stacking meta-learner
   
   **Approximate sizes:**
   - `data.zip`: ~2.5 GB (compressed)
   - Extracted: ~4 GB

### Quick Start: Making Predictions

Once installed, you can load a pretrained model and make predictions:

```python
import torch
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms

# Load preprocessed data and models
data = pd.read_csv('data/processed_properties.csv')  # Your property data
best_model = torch.load('notebooks/best_fusion_model.pth')
kmeans = pickle.load(open('notebooks/kmeans_model.pkl', 'rb'))

# Example: Predict price for a single property
property_row = data.iloc[0]  # Get first property

# Prepare tabular features
tabular_features = torch.FloatTensor([
    property_row['sqft_living'],
    property_row['grade'],
    property_row['bedrooms'],
    # ... (all required features)
])

# Load satellite image and extract ResNet50 features
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
image = Image.open(f'data/property_images/{property_row["id"]}.jpg')
image_tensor = transform(image).unsqueeze(0)

# Get prediction
best_model.eval()
with torch.no_grad():
    log_price_pred = best_model(tabular_features, image_tensor)
    price_pred = np.expm1(log_price_pred.item())  # Reverse log1p transformation

print(f"Predicted Price: ${price_pred:,.0f}")
```

### Project Structure

```
satellite-property-valuation/
├── README.md                          # This file
├── PROJECT_REPORT.md                  # Detailed technical report
├── requirements.txt                   # Python dependencies
├── data_fetcher.py                    # Satellite image download utility
│
├── notebooks/
│   ├── preprocessing.ipynb            # Data cleaning & feature engineering
│   ├── model_training.ipynb           # Model training & evaluation
│   ├── best_fusion_model.pth          # Pretrained PyTorch fusion model
│   ├── xgb_final.json                 # XGBoost tabular expert
│   ├── lgb_final.pkl                  # LightGBM tabular expert
│   ├── meta_learner_final.pkl         # Stacking meta-learner
│   ├── kmeans_model.pkl               # Spatial clustering (k=20)
│   ├── scaler_stack.pkl               # Feature scaler
│   ├── label_encoder.pkl              # Categorical encoder
│   └── ...                            # Other preprocessed artifacts
│
├── results/
│   ├── 23118016_final.csv             # Validation predictions
│   └── price_heatmap.html             # Interactive King County price map

```

---

## Workflow Overview

### 1. **Preprocessing Pipeline** (`preprocessing.ipynb`)

- **Data Cleaning**: Remove rows with missing values, invalid sqft/bedrooms
- **Target Transform**: Apply `log1p(price)` to reduce right-skew (4.03 → 0.41)
- **Feature Engineering**:
  - `lot_to_living_ratio`: Lot efficiency
  - `sqft_per_room`: Room density
  - `age`: Years since construction (2024 − year_built)
  - `dist_to_downtown`: Geodesic distance to Seattle center
  - `cluster_id`: K-Means (k=20) spatial cluster assignment
- **Scaling/Encoding**:
  - StandardScaler on continuous features (fit on train only)
  - LabelEncoder for cluster IDs
  - TargetEncoder for categorical variables (baseline track)
- **Visual Features**: ResNet50 (pretrained, frozen) → 512-d embeddings stored in `image_features.npy`

### 2. **Model Training** (`model_training.ipynb`)

**Tabular Experts:**
- **XGBoost**: ~3000 trees, lr=0.01, max_depth=8, early stopping
- **LightGBM**: 3000 trees (leaf-wise), num_leaves=45, lr=0.005
- **Shallow MLP**: 2-layer network with BatchNorm + GELU activation

**Fusion Network:**
- Concatenates standardized tabular + 512-d visual embeddings
- Parallel expert predictions + disagreement features (pairwise deltas)
- **Meta-Learner**: GradientBoostingRegressor arbitrates conflicts

**Optimization:**
- Loss: Huber (robust to outliers)
- Optimizer: AdamW
- LR Schedule: OneCycleLR
- Early Stopping: On validation loss

### 3. **Evaluation & Explainability**

- **SHAP Summary**: Top features by impact (grade, sqft_living, distance, waterfront)
- **Grad-CAM**: Visual attributions highlighting relevant regions in satellite tiles
- **Residual Analysis**: Histogram, Q-Q plots, and RMSE by price quantile

---

## Configuration & Usage

### Downloading Satellite Images

The `data_fetcher.py` script automates satellite image retrieval from Mapbox:

```python
python data_fetcher.py
```

**Before running**, configure:
```python
API_KEY = "your_mapbox_api_key_here"  # Get from https://account.mapbox.com/
BASE_DATA_PATH = "data/train.csv"     # Path to property data
ZOOM_LEVEL = 17                       # Zoom level (17 = ~500m per tile)
THREADS = 10                          # Concurrent downloads
```

### Loading Pretrained Models

```python
import torch
import pickle
import xgboost as xgb
from lightgbm import Booster

# Load all artifacts
fusion_model = torch.load('notebooks/best_fusion_model.pth')
xgb_expert = xgb.Booster(model_file='notebooks/xgb_final.json')
lgb_expert = Booster(model_name='lgb_final.pkl')
meta_learner = pickle.load(open('notebooks/meta_learner_final.pkl', 'rb'))
kmeans = pickle.load(open('notebooks/kmeans_model.pkl', 'rb'))
```

---

## Results & Insights

### Performance by Segment

- **Waterfront Properties**: Multimodal model captures water adjacency, tree density → +15% accuracy
- **Urban Cores**: Dense block patterns, street canopy → +12% accuracy
- **Suburban Areas**: Tabular features sufficient → +3% accuracy
- **Case Study**: A $3.2M waterfront property was undervalued by $1.76M by tabular baseline; multimodal corrected to $0.58M error

### Key Findings

1. **Visual dominates in micro-markets**: SHAP importance increases for gradient/texture features in high-variance neighborhoods
2. **Disagreement features matter**: XGB–NN delta inclusion improved meta-learner R² by ~0.003
3. **Frozen embeddings suffice**: Fine-tuning ResNet50 improved R² by <0.001 but increased training time 3×

---

## Documentation & Support

- **Technical Deep Dive**: See [PROJECT_REPORT.md](PROJECT_REPORT.md) for architecture diagrams, SHAP analysis, and Grad-CAM visualizations
- **Jupyter Notebooks**: 
  - [preprocessing.ipynb](notebooks/preprocessing.ipynb) – Data exploration & engineering
  - [model_training.ipynb](notebooks/model_training.ipynb) – Training loops & evaluation
- **Interactive Visualizations**: [price_heatmap.html](results/price_heatmap.html) – King County price distribution

---

## Contributing

We welcome contributions! To get started:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-improvement`
3. **Make changes** and test thoroughly
4. **Commit** with clear messages: `git commit -m "Add feature X"`
5. **Push** and open a **Pull Request**

### Guidelines
- Follow PEP 8 style (use `black` for formatting)
- Add docstrings to new functions
- Update [PROJECT_REPORT.md](PROJECT_REPORT.md) if adding major features
- Test on CPU and GPU if possible

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **ImportError: No module named 'torch'** | Run `pip install -r requirements.txt` |
| **CUDA out of memory** | Reduce batch_size in notebooks or use CPU (`device='cpu'`) |
| **Missing satellite images** | Check `logs/download_errors.log` for API failures; verify Mapbox API key |
| **Model weights not found** | Ensure `notebooks/` folder is in the same directory; check file paths |

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use M-REVS in your research or application, please cite:

```bibtex
@project{mrevs2024,
  title={M-REVS: Multimodal Real Estate Valuation System},
  author={Pokhariya, Astha},
  year={2024},
  url={https://github.com/astha156369/satellite-property-valuation}
}
```

---

## Contact & Maintainer

- **Author**: [Astha Pokhariya](https://github.com/astha156369)
- **Email**: asthapok106@gmail.com
- **LinkedIn**: [Astha Pokhariya](https://www.linkedin.com/in/astha-pokhariya-64654a280/)

---

## Acknowledgments

- **Dataset**: King County housing records with Mapbox satellite imagery
- **Frameworks**: PyTorch, scikit-learn, XGBoost, LightGBM
- **Inspiration**: Research on omitted-variable bias in real estate AVMs

---

**Last Updated**: January 2026 | **Status**: Active Development

# MANGO Demand Forecasting

A machine learning system for predicting weekly demand for fashion products using LightGBM ensemble models.

## 👤 Authors
Jan Aguiló, Júlia López, Tània Pazos, and Aniol Petit

## 🎯 Overview

This project implements a demand forecasting system for fashion products that:

- Predicts weekly sales for each product
- Aggregates predictions to total production requirements
- Uses ensemble learning for robust predictions
- Includes advanced feature engineering
- Provides an interactive Streamlit interface

## 📁 Project Structure

```
mango-demand-forecasting/
├── src/                          # Source code modules
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Data loading and preparation
│   ├── features/                 # Feature engineering
│   │   ├── __init__.py
│   │   └── engineering.py        # Binning and transformations
│   ├── models/                   # Model training and prediction
│   │   ├── __init__.py
│   │   ├── trainer.py            # Model training functions
│   │   └── predictor.py          # Prediction functions
│   └── utils/                    # Utilities and configuration
│       ├── __init__.py
│       └── config.py             # Configuration settings
├── notebooks/                    # Jupyter notebooks
│   └── data_exploration.ipynb    # Data preprocessing notebook
├── data/                         # Data files
│   ├── train.csv                 # Raw training data
│   ├── test.csv                  # Raw test data
│   └── sample_submission.csv     # Submission format
├── models/                       # Saved models
│   └── checkpoints/              # Model checkpoints
├── train.py                      # Main training script
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Installation

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd mango-demand-forecasting
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Step 1: Data Preprocessing

Run the Jupyter notebook to preprocess the raw data:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

This notebook will:
- Clean and transform the raw data
- Engineer features (clustering, trend features, etc.)
- Create `data/processed/train_processed.csv` and `data/processed/test_processed.csv`

### Step 2: Train Models

Train the ensemble model:

```bash
python train.py
```

This will:
- Load processed training data
- Perform time-based cross-validation
- Optionally tune hyperparameters (if enabled)
- Train an ensemble of 5 LightGBM models
- Save models to `models/checkpoints/`
- Generate predictions on test data
- Save results to `outputs/predictions.csv` and `outputs/feature_importance.csv`

### Step 3: Use Streamlit App

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

The app provides:
- Model overview and statistics
- Interactive prediction generation
- Analytics and insights
- Downloadable predictions

## 🔧 How It Works

### 1. Data Preprocessing

The preprocessing pipeline (`src/data/preprocessing.py`):

- **Loads** processed CSV files
- **Bins** numeric features (num_stores, num_sizes) to prevent overfitting
- **Identifies** categorical features
- **Handles** missing values and category alignment
- **Returns** feature matrices and target variables

### 2. Feature Engineering

Key features include:

- **Binned Features**: `num_stores_bin`, `num_sizes_bin` (categorical bins)
- **Cluster Features**: `emb_cluster`, `color_cluster` (embedding and color clusters)
- **Trend Features**: `velocity_1_3`, `trend_score`, `sim_to_top`, `sim_to_bottom`
- **Seasonality**: `is_week_23`, `is_black_friday`
- **Cluster Aggregations**: Cluster-level demand statistics
- **Similarity Features**: Similar product demand patterns
- **Family Features**: Family-level trend statistics

### 3. Model Training

The training pipeline (`src/models/trainer.py`):

1. **Time-based CV Split**: Splits data by seasons (86-88 for train, 89 for validation)
2. **Hyperparameter Tuning** (optional): Uses Optuna to find best parameters
3. **Ensemble Training**: Trains 5 models with different random seeds
4. **Feature Importance**: Extracts and saves feature importance

### 4. Prediction Pipeline

The prediction process (`src/models/predictor.py`):

1. **Weekly Predictions**: Model predicts sales for each product-week combination
2. **Penalization**: Applies a factor (default 1.09) to avoid stockouts
3. **Aggregation**: Sums weekly predictions to get total production per product
4. **Post-processing**: Ensures non-negative integers, handles edge cases

### 5. Model Architecture

- **Algorithm**: LightGBM (Gradient Boosting Decision Trees)
- **Ensemble**: 5 models averaged together
- **Validation**: Time-based cross-validation (no data leakage)
- **Objective**: Regression (MSE loss)

## ⚙️ Configuration

Edit `src/utils/config.py` to customize:

```python
# Model Configuration
PENALIZATION_FACTOR = 1.09  # Increase predictions by 9%
N_ENSEMBLE_MODELS = 5       # Number of models in ensemble

# Hyperparameter Tuning
ENABLE_HYPERPARAM_TUNING = False  # Set to True to enable
HYPERPARAM_TUNING_TRIALS = 50     # Number of Optuna trials

# Default LightGBM Parameters
DEFAULT_LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    # ... more parameters
}
```

## 🎨 Streamlit App

The Streamlit app (`app.py`) provides:

### Overview Tab
- Model statistics (samples, features, etc.)
- Feature importance visualization
- Model information

### Predictions Tab
- Interactive prediction generation
- Configurable penalization factor
- Downloadable CSV predictions
- Prediction preview

### Analytics Tab
- Production distribution charts
- Top/bottom products analysis
- Statistical summaries

### Info Tab
- Project documentation
- Technical details
- Usage instructions

## 📊 Features

### Advanced Feature Engineering

1. **Embedding Clustering**: Groups products by visual similarity using image embeddings
2. **Color Clustering**: Groups products by color using k-means
3. **Trend Analysis**: Identifies products similar to top/bottom performers
4. **Velocity Features**: Early sales velocity indicators
5. **Seasonality**: Captures seasonal patterns (week 23, Black Friday)
6. **Cluster Aggregations**: Historical performance by cluster
7. **Family Trends**: Family-level demand patterns

### Model Features

- **Ensemble Learning**: 5 models for robust predictions
- **Time-based Validation**: Prevents data leakage
- **Feature Selection**: Excludes low-importance features
- **Categorical Handling**: Proper handling of categorical features
- **Production Forecasting**: Aggregates weekly to total production

## 📝 Output Files

After running `train.py`, you'll get:

- `outputs/predictions.csv`: Final production predictions (ID, Production)
- `outputs/feature_importance.csv`: Feature importance rankings
- `outputs/weekly_predictions_debug.csv`: Weekly predictions breakdown (if enabled)
- `models/checkpoints/model_*.pkl`: Saved model files

---

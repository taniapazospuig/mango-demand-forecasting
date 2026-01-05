# MANGO Demand Forecasting

Mango demand forecasting solution for FME Datathon 2025. This project predicts demand (maximum potential sales with infinite stock) for fashion products in a new season, providing production quantities to send to suppliers. The solution uses historical data from four seasons (two full years plus validation), containing product image embeddings, product family and attributes, phase information, number of stores and sizes, and past sales and production data. An ensemble of LightGBM models is trained to forecast demand, with the evaluation metric penalizing lost sales more heavily than excess stock, reflecting real-world retail constraints.

## Authors

Jan Aguiló, Júlia López, Tània Pazos, and Aniol Petit

## Project Structure

```
mango-demand-forecasting/
├── src/                          # Source code modules
│   ├── data/                     # Data loading and preparation
│   │   └── preprocessing.py      # Functions to load processed CSVs for ML
│   ├── models/                   # Model training and prediction
│   │   ├── trainer.py            # LightGBM training, CV, hyperparameter tuning
│   │   └── predictor.py          # Production prediction functions
│   └── utils/                    # Configuration
│       └── config.py             # Model parameters and settings
├── data/                         # Data files
│   ├── train.csv                 # Raw training data
│   ├── test.csv                  # Raw test data
│   └── processed/                # Processed datasets (created after preprocessing)
│       ├── train_processed.csv   # Feature-engineered training data
│       └── test_processed.csv    # Feature-engineered test data
├── models/                       # Saved models
│   └── checkpoints/              # Trained model files (model_0.pkl, model_1.pkl, ...)
├── outputs/                      # Predictions and feature importance
├── data_preprocessing.ipynb      # Feature engineering notebook (creates processed CSVs)
├── train.py                      # Main training script
└── requirements.txt              # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mango-demand-forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Feature Engineering (One-Time)

Run the Jupyter notebook to perform feature engineering on the raw data:

```bash
jupyter notebook data_preprocessing.ipynb
```

**What this notebook does:**
- Loads raw data from `data/train.csv` and `data/test.csv`
- Performs extensive feature engineering:
  - Color clustering (KMeans on color names)
  - Image embedding PCA and clustering (83 PCA components, 22 clusters)
  - Trend features (similarity to top/bottom performers)
  - Cluster-level aggregations (velocity, demand trends, season-on-season growth)
  - Family-level aggregations (velocity, demand trends)
  - Temporal features (previous season statistics)
  - Seasonality indicators (Black Friday, week 23)
- Saves processed datasets to:
  - `data/processed/train_processed.csv`
  - `data/processed/test_processed.csv`

**Note:** This step is computationally intensive and only needs to be run once (or when you want to regenerate features).

### Step 2: Train Models

Train the ensemble model using the processed data:

```bash
python train.py
```

**What `train.py` does:**
1. **Data Loading** (`src/data/preprocessing.py`):
   - Loads processed training data from `data/processed/train_processed.csv`
   - Separates features (X) from target (y = weekly_demand)
   - Identifies categorical features for LightGBM
   - Extracts metadata (seasons, bin_info)

2. **Cross-Validation** (`src/models/trainer.py`):
   - Performs time-based cross-validation (seasons 86-88 for training, 89 for validation)

3. **Hyperparameter Tuning** (optional):
   - Runs Optuna optimization if enabled in `src/utils/config.py`

4. **Model Training** (`src/models/trainer.py`):
   - Trains an ensemble of 5 LightGBM models with different random seeds (42-46)
   - Saves models to `models/checkpoints/model_*.pkl`

5. **Prediction** (`src/models/predictor.py`):
   - Loads and processes test data from `data/processed/test_processed.csv`
   - Limits to max 30 weeks per product
   - Makes ensemble predictions and aggregates weekly predictions to total production
   - Applies penalization factor (1.09) to avoid stockouts
   - Saves results to `outputs/predictions.csv` and `outputs/feature_importance.csv`

**Note:** You can run `train.py` multiple times without re-running the preprocessing notebook, as long as the processed CSVs exist.

## Model Architecture

- **Algorithm**: LightGBM (Gradient Boosting Decision Trees)
- **Ensemble**: 5 models averaged together (different random seeds: 42-46)
- **Validation**: Time-based cross-validation (seasons 86-88 for train, 89 for validation)
- **Objective**: Regression (MSE loss)
- **Training**: Models trained on all training data (seasons 86-89) after CV evaluation
- **Boosting Rounds**: 1000 iterations per model
- **Penalization Factor**: 1.09 (predictions multiplied by 9% to avoid stockouts)

## Key Features

The model uses extensive feature engineering including:

- **Image Embedding Clustering**: Identifies visually similar products using embedding clusters
- **Color Clustering**: Groups products by color similarity for trend analysis
- **Temporal Features**: Previous season aggregations (cluster demand last season, family velocity last season) to capture temporal patterns
- **Seasonality Indicators**: Black Friday detection, week 23 features
- **Aggregations**: Cluster-level and family-level demand aggregations and statistics
- **Binned Features**: Categorical bins for `num_stores` and `num_sizes` to handle numeric features
- **Trend Analysis**: Similarity to top/bottom performers, cluster velocity, demand trends
- **Weekly Forecasting**: Predicts weekly demand per product, then aggregates to total production requirements

## Code Organization

The project is organized into modules:

- **`src/data/preprocessing.py`**: Functions to load processed CSVs and prepare them for ML (separate features/target, identify categoricals, extract metadata)
- **`src/models/trainer.py`**: Model training functions (LightGBM training, time-based CV split, hyperparameter tuning, feature importance)
- **`src/models/predictor.py`**: Prediction functions (ensemble prediction, aggregation to production quantities)
- **`src/utils/config.py`**: Configuration settings

## Configuration

Edit `src/utils/config.py` to customize:

- **PENALIZATION_FACTOR**: Multiplier for predictions to avoid stockouts (default: 1.09)
- **N_ENSEMBLE_MODELS**: Number of models in ensemble (default: 5)
- **ENABLE_HYPERPARAM_TUNING**: Enable Optuna hyperparameter optimization (default: False)
- **HYPERPARAM_TUNING_TRIALS**: Number of Optuna trials if tuning enabled (default: 50)
- **DEFAULT_LIGHTGBM_PARAMS**: Default hyperparameters for LightGBM

## Output Files

After running `train.py`, you'll get:

- `outputs/predictions.csv`: Final production predictions with columns (ID, Production)
- `outputs/feature_importance.csv`: Feature importance rankings from the first model
- `models/checkpoints/model_0.pkl` through `model_4.pkl`: Saved ensemble model files

**Note**: The predictions file includes all test product IDs. Products with missing predictions are assigned Production = 0.

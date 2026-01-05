# MANGO Demand Forecasting

Mango demand forecasting solution for FME Datathon 2025. This project predicts demand (maximum potential sales with infinite stock) for fashion products in a new season, providing production quantities to send to suppliers. The solution uses historical data from four seasons (two full years plus validation), containing product image embeddings, product family and attributes, phase information, number of stores and sizes, and past sales and production data. An ensemble of LightGBM models is trained to forecast demand, with the evaluation metric penalizing lost sales more heavily than excess stock, reflecting real-world retail constraints.

## Authors

Jan Aguiló, Júlia López, Tània Pazos, and Aniol Petit

## Project Structure

```
mango-demand-forecasting/
├── src/                          # Source code modules
│   ├── utils/                    # Configuration
│       └── config.py             # Model parameters and settings
├── data/                         # Data files
│   └── processed/                # Processed datasets (created after preprocessing)
├── models/                       # Saved models
│   └── checkpoints/              # Trained model files (model_0.pkl, model_1.pkl, ...)
├── outputs/                      # Predictions and feature importance
├── data_preprocessing.ipynb      # Data preprocessing notebook
├── train.py                      # Training script
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

### Step 1: Data Preprocessing

Run the Jupyter notebook to preprocess the raw data:

```bash
jupyter notebook data_preprocessing.ipynb
```

This notebook cleans and transforms the raw data, engineers features (clustering, trend features, etc.), and creates `data/processed/train_processed.csv` and `data/processed/test_processed.csv`.

### Step 2: Train Models

Train the ensemble model:

```bash
python train.py
```

This will:
- Load processed training data from `data/processed/train_processed.csv`
- Perform time-based cross-validation (seasons 86-88 for training, 89 for validation)
- Optionally run hyperparameter tuning (if enabled in config)
- Train an ensemble of 5 LightGBM models with different random seeds
- Save models to `models/checkpoints/model_*.pkl`
- Process test data from `data/processed/test_processed.csv` (max 30 weeks per product)
- Generate predictions using the ensemble with a penalization factor of 1.09
- Save results to `outputs/predictions.csv` and `outputs/feature_importance.csv`

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

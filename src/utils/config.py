"""Configuration settings for the mango demand forecasting model."""

# Model Configuration
PENALIZATION_FACTOR = 1.09  # Increase predictions by 9% to avoid being short on demand
N_ENSEMBLE_MODELS = 5  # Number of models to ensemble (different random seeds)

# Hyperparameter Tuning
ENABLE_HYPERPARAM_TUNING = False  # Set to True to enable hyperparameter tuning
HYPERPARAM_TUNING_TRIALS = 50  # Number of trials for Optuna

# Data Paths
DATA_DIR = "data"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models/checkpoints"
OUTPUT_DIR = "outputs"

# Feature Configuration
LOW_IMPORTANCE_FEATURES = [
    'cluster_velocity_1_6',
    'cluster_peak_week',
    'family_demand_trend',
    'cluster_popularity',
    'cluster_demand_slope',
    'cluster_season_growth',
    'aggregated_family',
    'cluster_yoy_change',
]

EXCLUDED_COLUMNS = [
    'color_name', 'image_embedding', 'embedding_array', 'ID', 'knit_structure',
    'num_week_iso', 'weekly_demand', 'Production', 'weekly_sales',
    'waist_type', 'is_fall', 'category', 'id_season', 'moment', 'woven_structure',
    'R', 'G', 'B', 'phase_in', 'phase_out', 'num_stores', 'num_sizes'
]

# Default LightGBM Parameters
DEFAULT_LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}


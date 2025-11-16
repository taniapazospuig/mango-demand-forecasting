"""
Mango Weekly Demand Forecast Model - CatBoost Pipeline
CatBoost regression model to forecast weekly sales for fashion products
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
import optuna
import warnings
warnings.filterwarnings('ignore')

# Configuration
PENALIZATION_FACTOR = 1.09
N_ENSEMBLE_MODELS = 5

# ---------------------------------------------------------------------------------------
# 1. Load & Prepare Training Data
# ---------------------------------------------------------------------------------------

def load_and_prepare_train_data(filepath='train_processed.csv', return_seasons=False):
    print("Loading training data with CatBoost preprocessing...")
    
    df = pd.read_csv(filepath)

    exclude_cols = [
        'color_name', 'image_embedding', 'embedding_array', 'ID',
        'knit_structure', 'num_week_iso', 'weekly_demand',
        'Production', 'weekly_sales'
    ]
    
    all_features = [col for col in df.columns if col not in exclude_cols]

    # Identify categorical features (object dtype)
    categorical_cols = df[all_features].select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert categorical fields to string (CatBoost requirement)
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("MISSING")

    # Target
    y = df["weekly_sales"].copy()
    X = df[all_features].copy()

    if return_seasons:
        return X, y, categorical_cols, df["id_season"].copy()
    else:
        return X, y, categorical_cols

# ---------------------------------------------------------------------------------------
# 2. Preprocess Test Data
# ---------------------------------------------------------------------------------------

def process_test_data(filepath='test_processed.csv', max_weeks=30, train_categorical_cols=None):
    print("Loading test data...")
    df_test = pd.read_csv(filepath)

    if max_weeks is None:
        max_weeks = 30

    rows = []
    for _, row in df_test.iterrows():
        if ('life_cycle_length' in row) and pd.notna(row['life_cycle_length']):
            num_weeks = max(1, min(int(row['life_cycle_length']), max_weeks))
        else:
            num_weeks = max_weeks

        for w in range(1, num_weeks + 1):
            r = row.to_dict()
            r["weeks_since_launch"] = w
            rows.append(r)

    df_processed = pd.DataFrame(rows)

    # Convert bools to int
    if 'has_plus_sizes' in df_processed.columns:
        df_processed['has_plus_sizes'] = df_processed['has_plus_sizes'].astype(int)

    exclude_cols = [
        'color_name', 'image_embedding', 'embedding_array', 'ID',
        'knit_structure', 'num_week_iso', 'weekly_demand',
        'Production', 'weekly_sales'
    ]

    all_features = [col for col in df_processed.columns if col not in exclude_cols]

    # Process categorical columns
    categorical_cols_test = df_processed[all_features].select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols_test:
        df_processed[col] = df_processed[col].astype(str).fillna("MISSING")

    X_test = df_processed[all_features].copy()
    df_meta = df_processed[['ID', 'weeks_since_launch']].copy()

    return X_test, df_meta

# ---------------------------------------------------------------------------------------
# 3. CatBoost Model Training
# ---------------------------------------------------------------------------------------

def train_catboost_model(X_train, y_train, X_val=None, y_val=None,
                         categorical_features=None, params=None):

    if params is None:
        params = {
            "loss_function": "RMSE",
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 3,
            "random_seed": 42,
            "verbose": False
        }

    train_pool = Pool(X_train, y_train, cat_features=categorical_features)

    if X_val is not None:
        val_pool = Pool(X_val, y_val, cat_features=categorical_features)
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=False)
    else:
        model = CatBoostRegressor(**params)
        model.fit(train_pool, verbose=False)

    return model

# ---------------------------------------------------------------------------------------
# 4. CatBoost Hyperparameter Tuning with Optuna
# ---------------------------------------------------------------------------------------

def hyperparameter_tuning_catboost(X_train, y_train, X_val, y_val, categorical_features, n_trials=50):

    print("\nStarting Optuna tuning for CatBoost...")

    def objective(trial):
        params = {
            "loss_function": "RMSE",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_seed": 42,
            "verbose": False
        }

        model = CatBoostRegressor(**params)
        train_pool = Pool(X_train, y_train, cat_features=categorical_features)
        val_pool = Pool(X_val, y_val, cat_features=categorical_features)

        model.fit(train_pool, eval_set=val_pool, verbose=False)

        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nBest CatBoost hyperparameters:")
    print(study.best_params)
    print("MSE:", study.best_value)

    return study.best_params

# ---------------------------------------------------------------------------------------
# 5. Predict Production
# ---------------------------------------------------------------------------------------

def predict_production(model, X_test, df_meta, penalization_factor=PENALIZATION_FACTOR):
    weekly_pred = model.predict(X_test)
    weekly_pred = weekly_pred * penalization_factor

    pred_df = df_meta.copy()
    pred_df["predicted_weekly_sales"] = weekly_pred

    production_df = (
        pred_df.groupby("ID")["predicted_weekly_sales"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_weekly_sales": "Production"})
    )

    production_df["Production"] = (
        production_df["Production"].clip(lower=0).round().astype(int)
    )

    return production_df

# ---------------------------------------------------------------------------------------
# 6. Main Pipeline
# ---------------------------------------------------------------------------------------

def main():
    print("="*60)
    print("Mango Weekly Forecasting – CatBoost Model")
    print("="*60)

    # Load training data
    X, y, categorical_features, seasons = load_and_prepare_train_data(
        'train_processed.csv', return_seasons=True
    )

    # Time-based CV
    X_train_cv = X[seasons.isin([86,87,88])]
    y_train_cv = y[seasons.isin([86,87,88])]
    X_val_cv = X[seasons == 89]
    y_val_cv = y[seasons == 89]

    # Hyperparameter tuning
    best_params = hyperparameter_tuning_catboost(
        X_train_cv, y_train_cv, X_val_cv, y_val_cv,
        categorical_features,
        n_trials=40
    )

    # Train ensemble of CatBoost models
    models = []
    for i in range(N_ENSEMBLE_MODELS):
        print(f"Training CatBoost model {i+1}/{N_ENSEMBLE_MODELS}...")
        params = best_params.copy()
        params["random_seed"] = 42 + i

        model = train_catboost_model(
            X, y,
            categorical_features=categorical_features,
            params=params
        )
        models.append(model)

    # Process test data
    X_test, df_meta = process_test_data('test_processed.csv', max_weeks=30)

    # Predict using ensemble
    print("Predicting production...")
    preds = np.mean([m.predict(X_test) for m in models], axis=0)
    preds = preds * PENALIZATION_FACTOR

    pred_df = df_meta.copy()
    pred_df["predicted_weekly_sales"] = preds
    production_df = pred_df.groupby("ID")["predicted_weekly_sales"].sum().reset_index()
    production_df["Production"] = production_df["predicted_weekly_sales"].clip(lower=0).round().astype(int)
    production_df = production_df[["ID", "Production"]]

    production_df.to_csv("predictions_catboost.csv", index=False)
    print("Saved predictions to predictions_catboost.csv")

if __name__ == "__main__":
    main()

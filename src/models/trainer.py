"""Model training functions for LightGBM."""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error

# Import early_stopping callback (compatible with different LightGBM versions)
try:
    from lightgbm import early_stopping
except ImportError:
    try:
        from lightgbm.callback import early_stopping
    except ImportError:
        early_stopping = None


def time_based_cv_split(X, y, seasons, train_seasons, val_season):
    """
    Split data by season for time-based cross-validation.
    
    Args:
        X: Feature dataframe
        y: Target array
        seasons: Array of season IDs
        train_seasons: List of season IDs for training
        val_season: Season ID for validation
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    train_mask = np.isin(seasons, train_seasons)
    val_mask = (seasons == val_season)
    
    X_train = X[train_mask].copy()
    X_val = X[val_mask].copy()
    y_train = y[train_mask]
    y_val = y[val_mask]
    
    print(f"Train set: {len(X_train):,} samples (seasons {train_seasons})")
    print(f"Val set: {len(X_val):,} samples (season {val_season})")
    
    return X_train, X_val, y_train, y_val


def train_lightgbm_model(X, y, categorical_features, params, num_boost_round=1000, verbose_eval=100):
    """
    Train a LightGBM model.
    
    Args:
        X: Feature dataframe
        y: Target array
        categorical_features: List of categorical feature names
        params: LightGBM parameters
        num_boost_round: Number of boosting rounds
        verbose_eval: Verbosity level for evaluation
        
    Returns:
        Trained LightGBM model
    """
    # Prepare categorical features
    categorical_feature_indices = []
    if categorical_features:
        for cat_feat in categorical_features:
            if cat_feat in X.columns:
                idx = X.columns.get_loc(cat_feat)
                categorical_feature_indices.append(idx)
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(
        X.values,
        label=y,
        categorical_feature=categorical_feature_indices if categorical_feature_indices else 'auto',
        free_raw_data=False
    )
    
    # Train model
    callbacks = []
    if verbose_eval > 0 and early_stopping is not None:
        callbacks.append(early_stopping(stopping_rounds=50, verbose=False))
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data],
        valid_names=['train'],
        verbose_eval=verbose_eval,
        callbacks=callbacks if callbacks else None
    )
    
    return model


def hyperparameter_tuning(X_train, y_train, X_val, y_val, categorical_features, n_trials=50):
    """
    Perform hyperparameter tuning using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        categorical_features: List of categorical feature names
        n_trials: Number of Optuna trials
        
    Returns:
        Best hyperparameters dictionary
    """
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbose': -1,
            'random_state': 42
        }
        
        # Prepare categorical features
        categorical_feature_indices = []
        if categorical_features:
            for cat_feat in categorical_features:
                if cat_feat in X_train.columns:
                    idx = X_train.columns.get_loc(cat_feat)
                    categorical_feature_indices.append(idx)
        
        train_data = lgb.Dataset(
            X_train.values,
            label=y_train,
            categorical_feature=categorical_feature_indices if categorical_feature_indices else 'auto',
            free_raw_data=False
        )
        
        val_data = lgb.Dataset(
            X_val.values,
            label=y_val,
            categorical_feature=categorical_feature_indices if categorical_feature_indices else 'auto',
            reference=train_data,
            free_raw_data=False
        )
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            valid_names=['val'],
            verbose_eval=False,
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)] if early_stopping is not None else None
        )
        
        y_pred = model.predict(X_val.values)
        mse = mean_squared_error(y_val, y_pred)
        
        return mse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params['objective'] = 'regression'
    best_params['metric'] = 'mse'
    best_params['boosting_type'] = 'gbdt'
    best_params['verbose'] = -1
    best_params['random_state'] = 42
    
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best validation MSE: {study.best_value:.4f}")
    
    return best_params


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    importance = model.feature_importance(importance_type='gain')
    
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    return feature_imp_df


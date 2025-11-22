"""Model training functions including LightGBM training and hyperparameter tuning."""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna
from typing import List, Tuple, Optional, Dict
from ..utils.config import DEFAULT_LIGHTGBM_PARAMS, ENABLE_HYPERPARAM_TUNING, HYPERPARAM_TUNING_TRIALS


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: List[str],
    params: Optional[Dict] = None,
    num_boost_round: int = 1000,
    valid_sets: Optional[List[Tuple]] = None,
    verbose_eval: int = 100
) -> lgb.Booster:
    """
    Train LightGBM regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target variable
    categorical_features : list
        List of categorical feature names
    params : dict, optional
        LightGBM parameters (defaults to DEFAULT_LIGHTGBM_PARAMS)
    num_boost_round : int
        Number of boosting rounds
    valid_sets : list, optional
        List of (X_val, y_val) tuples for validation
    verbose_eval : int
        Verbosity level for evaluation
        
    Returns:
    --------
    lgb.Booster
        Trained LightGBM model
    """
    if params is None:
        params = DEFAULT_LIGHTGBM_PARAMS.copy()
    
    # Get categorical feature indices
    cat_indices = [i for i, col in enumerate(X_train.columns) 
                   if col in categorical_features]
    
    # Create LightGBM dataset with categorical features
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices)
    
    # Prepare validation sets if provided
    callbacks = []
    if valid_sets:
        valid_datasets = []
        for X_val, y_val in valid_sets:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, 
                                  categorical_feature=cat_indices)
            valid_datasets.append(val_data)
        callbacks.append(lgb.early_stopping(stopping_rounds=50))
        callbacks.append(lgb.log_evaluation(period=verbose_eval))
    else:
        callbacks.append(lgb.log_evaluation(period=verbose_eval))
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_datasets if valid_sets else None,
        callbacks=callbacks
    )
    
    return model


def time_based_cv_split(
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    train_seasons: List[int] = [86, 87, 88],
    val_season: int = 89
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data based on seasons for cross-validation.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    seasons : pd.Series
        Season IDs
    train_seasons : list
        List of season IDs to use for training
    val_season : int
        Season ID to use for validation
        
    Returns:
    --------
    Tuple of (X_train, X_val, y_train, y_val)
    """
    train_mask = seasons.isin(train_seasons)
    val_mask = seasons == val_season
    
    X_train = X[train_mask].copy()
    X_val = X[val_mask].copy()
    y_train = y[train_mask].copy()
    y_val = y[val_mask].copy()
    
    print(f"CV Split - Train: {len(X_train):,} samples (seasons {train_seasons}), "
          f"Val: {len(X_val):,} samples (season {val_season})")
    
    return X_train, X_val, y_train, y_val


def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_features: List[str],
    n_trials: int = 100
) -> Dict:
    """
    Hyperparameter tuning using Optuna.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target variable
    X_val : pd.DataFrame
        Validation feature matrix
    y_val : pd.Series
        Validation target variable
    categorical_features : list
        List of categorical feature names
    n_trials : int
        Number of Optuna trials
        
    Returns:
    --------
    dict
        Best hyperparameters found
    """
    print(f"\nStarting hyperparameter tuning with {n_trials} trials...")
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'verbose': -1,
            'random_state': 42
        }
        
        # Get categorical feature indices
        cat_indices = [i for i, col in enumerate(X_train.columns) 
                       if col in categorical_features]
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, 
                              categorical_feature=cat_indices)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        mse = mean_squared_error(y_val, y_pred)
        
        return mse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest hyperparameters found:")
    print(f"  MSE: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Build final params dict with best values
    best_params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        **study.best_params
    }
    
    return best_params


def get_feature_importance(model: lgb.Booster, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract and display feature importance.
    
    Parameters:
    -----------
    model : lgb.Booster
        Trained LightGBM model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importance sorted by importance
    """
    importance = model.feature_importance(importance_type='gain')
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance Ranking:")
    print(feature_imp_df.to_string(index=False))
    
    return feature_imp_df


"""Main training script for the Mango Demand Forecasting model."""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.data.preprocessing import load_and_prepare_train_data, process_test_data
from src.models.trainer import (
    train_lightgbm_model,
    time_based_cv_split,
    hyperparameter_tuning,
    get_feature_importance
)
from src.models.predictor import predict_production
from src.utils.config import (
    PENALIZATION_FACTOR,
    N_ENSEMBLE_MODELS,
    ENABLE_HYPERPARAM_TUNING,
    HYPERPARAM_TUNING_TRIALS,
    DEFAULT_LIGHTGBM_PARAMS
)


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Mango Weekly Demand Forecast - Base Model Pipeline")
    print("=" * 60)
    
    # Load training data with seasons
    X, y, categorical_features, seasons, bin_info = load_and_prepare_train_data(
        'data/processed/train_processed.csv', return_seasons=True
    )
    print(f"Loaded {len(X):,} training samples")
    print(f"Features: {list(X.columns)}")
    
    # Step 1: Cross-validation split (seasons 86,87,88 vs 89)
    print("\n" + "=" * 60)
    print("Step 1: Cross-Validation Split")
    print("=" * 60)
    X_train_cv, X_val_cv, y_train_cv, y_val_cv = time_based_cv_split(
        X, y, seasons, train_seasons=[86, 87, 88], val_season=89
    )
    
    # Step 2: Hyperparameter tuning (optional)
    if ENABLE_HYPERPARAM_TUNING:
        print("\n" + "=" * 60)
        print("Step 2: Hyperparameter Tuning")
        print("=" * 60)
        best_params = hyperparameter_tuning(
            X_train_cv, y_train_cv, X_val_cv, y_val_cv, 
            categorical_features, n_trials=HYPERPARAM_TUNING_TRIALS
        )
    else:
        print("\n" + "=" * 60)
        print("Step 2: Using Default Hyperparameters")
        print("=" * 60)
        best_params = DEFAULT_LIGHTGBM_PARAMS.copy()
    
    # Step 3: Train ensemble of models on all training data
    print("\n" + "=" * 60)
    print(f"Step 3: Training Ensemble of {N_ENSEMBLE_MODELS} Models on All Training Data")
    print("=" * 60)
    
    models = []
    import pickle
    import os
    os.makedirs('models/checkpoints', exist_ok=True)
    
    for i in range(N_ENSEMBLE_MODELS):
        print(f"\nTraining model {i+1}/{N_ENSEMBLE_MODELS}...")
        model_params = best_params.copy()
        model_params['random_state'] = 42 + i
        
        model = train_lightgbm_model(
            X, y, categorical_features, 
            params=model_params,
            num_boost_round=1000,
            verbose_eval=100 if i == 0 else 0
        )
        models.append(model)
        
        # Save model
        model_path = f'models/checkpoints/model_{i}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✓ Model saved to {model_path}")
    
    # Feature importance
    feature_names = list(X.columns)
    feature_imp_df = get_feature_importance(models[0], feature_names)
    
    # Step 4: Process test data and make predictions
    print("\n" + "=" * 60)
    print("Step 4: Making Predictions on Test Data")
    print("=" * 60)
    
    X_test, df_meta = process_test_data(
        'data/processed/test_processed.csv',
        max_weeks=30,
        train_categorical_cols=categorical_features,
        bin_info=bin_info
    )
    
    # Ensure test features match training features
    for col in X.columns:
        if col not in X_test.columns:
            print(f"Warning: Missing column {col} in test data, filling with default")
            if col in categorical_features:
                if col in ['num_stores_bin', 'num_sizes_bin']:
                    most_common = X[col].value_counts().index[0]
                    X_test[col] = pd.Categorical([most_common] * len(X_test), categories=X[col].cat.categories)
                else:
                    X_test[col] = pd.Categorical(['MISSING'] * len(X_test))
            else:
                X_test[col] = 0
    
    # Reorder test columns to match training
    X_test = X_test[X.columns]
    
    # Ensure categorical columns have same categories as training
    for col in categorical_features:
        if col in X_test.columns and col in X.columns:
            train_cats = X[col].cat.categories
            test_cats = X_test[col].cat.categories
            all_cats = train_cats.union(test_cats)
            X_test[col] = X_test[col].cat.set_categories(all_cats)
            
            if col in ['num_stores_bin', 'num_sizes_bin']:
                if X_test[col].isna().any():
                    most_common = X[col].value_counts().index[0]
                    X_test[col] = X_test[col].fillna(most_common)
    
    # Predict production
    production_df = predict_production(
        models, X_test, df_meta, PENALIZATION_FACTOR, save_weekly_debug=True
    )
    
    # Ensure all test product IDs are in predictions
    test_ids = pd.read_csv('data/processed/test_processed.csv', usecols=['ID'])
    all_test_ids = test_ids['ID'].unique()
    predicted_ids = set(production_df['ID'].unique())
    
    missing_ids = set(all_test_ids) - predicted_ids
    if missing_ids:
        missing_df = pd.DataFrame({'ID': list(missing_ids), 'Production': 0})
        production_df = pd.concat([production_df, missing_df], ignore_index=True)
    
    # Sort by ID
    production_df = production_df.sort_values('ID').reset_index(drop=True)
    
    # Save predictions
    import os
    os.makedirs('outputs', exist_ok=True)
    output_file = 'outputs/predictions.csv'
    production_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    print(f"Total products: {len(production_df)}")
    print(f"Total predicted production: {production_df['Production'].sum():,}")
    print(f"Average production per product: {production_df['Production'].mean():.2f}")
    print(f"Min production: {production_df['Production'].min()}")
    print(f"Max production: {production_df['Production'].max()}")
    
    # Save feature importance
    feature_imp_df.to_csv('outputs/feature_importance.csv', index=False)
    print(f"\nFeature importance saved to outputs/feature_importance.csv")
    
    print("\n" + "=" * 60)
    print("Training and prediction completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


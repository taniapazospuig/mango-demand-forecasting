"""
Mango Weekly Demand Forecast Model - Base Pipeline
LightGBM regression model to forecast weekly sales for fashion products
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Configuration
PENALIZATION_FACTOR = 1.09  # Increase predictions by 7.5% to avoid being short on demand
N_ENSEMBLE_MODELS = 5  # Number of models to ensemble (different random seeds)

def load_and_prepare_train_data(filepath='train_processed.csv', return_seasons=False):
    """Load training data and select required features"""
    print("Loading training data...")
    df = pd.read_csv(filepath)
    
    # Create new features from num_week_iso
    if 'num_week_iso' in df.columns:
        df["is_week23_peak"] = (df["num_week_iso"] == 23).astype(int)
        # Black Friday is usually ISO week 47-48
        df["is_black_friday_week"] = df["num_week_iso"].isin([47, 48]).astype(int)
    
    # Create historic average demand by ID
    if 'ID' in df.columns and 'weekly_sales' in df.columns:
        avg_sales_ID = df.groupby("ID")["weekly_sales"].mean()
        df["avg_sales_ID"] = df["ID"].map(avg_sales_ID)
        # Fill NaN with 0 for products without historical data
        df["avg_sales_ID"] = df["avg_sales_ID"].fillna(0)
    
    # Create lag features grouped by ID and id_season
    if 'ID' in df.columns and 'id_season' in df.columns and 'weekly_sales' in df.columns:
        # Sort by ID, id_season, and num_week_iso to ensure proper ordering
        if 'num_week_iso' in df.columns:
            df = df.sort_values(['ID', 'id_season', 'num_week_iso']).reset_index(drop=True)
        
        # Create lag features
        df["lag1"] = df.groupby(["ID", "id_season"])["weekly_sales"].shift(1)
        df["lag2"] = df.groupby(["ID", "id_season"])["weekly_sales"].shift(2)
        df["lag3"] = df.groupby(["ID", "id_season"])["weekly_sales"].shift(3)
        df["lag4"] = df.groupby(["ID", "id_season"])["weekly_sales"].shift(4)
        
        # Compute 4-week moving average
        df["lag_mean4"] = df[["lag1", "lag2", "lag3", "lag4"]].mean(axis=1)
        
        # Fill NaN values with 0 for lag features
        df[["lag1", "lag2", "lag3", "lag4", "lag_mean4"]] = df[["lag1", "lag2", "lag3", "lag4", "lag_mean4"]].fillna(0)
    
    # Define columns to exclude
    # Note: emb_pca_*, emb_cluster, emb_dist are included as features
    exclude_cols = ['color_name', 'image_embedding', 'embedding_array', 'ID', 'knit_structure', 
                    'num_week_iso', 'weekly_demand', 'Production', 'weekly_sales',
                    'category', 'is_fall', 'waist_type', 'id_season', 'moment', 'woven_structure']
    
    # Get all features (all columns except excluded ones and target)
    all_features = [col for col in df.columns if col not in exclude_cols]
    
    # Identify object type columns (categorical features)
    # Note: CSV files load categorical columns as object dtype, so we check for object
    object_cols = df[all_features].select_dtypes(include=['object']).columns.tolist()
    
    # emb_cluster should be treated as categorical (even though it's numeric)
    # Add it to categorical features if it exists
    if 'emb_cluster' in all_features and 'emb_cluster' not in object_cols:
        # Fill any missing values with -1 (sentinel for invalid embeddings)
        if df['emb_cluster'].isna().any():
            df['emb_cluster'] = df['emb_cluster'].fillna(-1)
        # Convert emb_cluster to categorical type
        df['emb_cluster'] = df['emb_cluster'].astype(int).astype('category')
        object_cols.append('emb_cluster')
    
    # Count embedding features
    embedding_features = [col for col in all_features if col.startswith('emb_pca_')]
    print(f"Found {len(all_features)} features, {len(object_cols)} categorical columns")
    print(f"  Including {len(embedding_features)} PCA embedding features (emb_pca_1 to emb_pca_{len(embedding_features)})")
    if 'emb_cluster' in all_features:
        print(f"  Including emb_cluster (categorical)")
    if 'emb_dist' in all_features:
        print(f"  Including emb_dist (numeric)")
    
    # Convert object columns to category dtype and handle missing values
    for col in object_cols:
        if col in df.columns:
            # emb_cluster is already handled above, skip it here
            if col == 'emb_cluster':
                continue
            # Fill missing values with 'MISSING' before converting to category
            df[col] = df[col].astype(str).fillna('MISSING').astype('category')
    
    # Create feature matrix and target
    X = df[all_features].copy()
    y = df['weekly_sales'].copy()
    
    if return_seasons:
        seasons = df['id_season'].copy() if 'id_season' in df.columns else None
        return X, y, object_cols, seasons
    else:
        return X, y, object_cols

def process_test_data(filepath='test_processed.csv', max_weeks=None, train_categorical_cols=None, 
                      train_filepath='train_processed.csv'):
    """
    Process preprocessed test data to create rows for multiple weeks per product
    Uses life_cycle_length from each product to determine how many weeks to predict
    If life_cycle_length is missing, uses max_weeks (default 30) as fallback
    
    Args:
        train_filepath: Path to training data file to compute avg_sales_ID from historical data
    """
    print("Loading preprocessed test data...")
    df_test = pd.read_csv(filepath)
    
    # Compute avg_sales_ID from training data if available
    avg_sales_ID_map = None
    try:
        df_train = pd.read_csv(train_filepath)
        if 'ID' in df_train.columns and 'weekly_sales' in df_train.columns:
            avg_sales_ID_map = df_train.groupby("ID")["weekly_sales"].mean().to_dict()
            print(f"Computed avg_sales_ID from training data for {len(avg_sales_ID_map)} products")
    except Exception as e:
        print(f"Warning: Could not load training data for avg_sales_ID: {e}")
    
    # Use life_cycle_length if available, otherwise use max_weeks (default 30)
    if max_weeks is None:
        max_weeks = 30
    
    # Create weeks_since_launch for each product-week combination
    # Use each product's life_cycle_length if available, otherwise use max_weeks
    product_rows = []
    for _, row in df_test.iterrows():
        # Get the number of weeks for this product
        if 'life_cycle_length' in row and pd.notna(row['life_cycle_length']):
            num_weeks = int(row['life_cycle_length'])
            # Ensure it's at least 1 and not unreasonably large
            num_weeks = max(1, min(num_weeks, max_weeks))
        else:
            # Fallback to max_weeks if life_cycle_length is missing
            num_weeks = max_weeks
        
        # Get phase_in date if available for computing ISO week
        phase_in_date = None
        if 'phase_in' in row and pd.notna(row['phase_in']):
            try:
                # Try to parse the date (handle different formats)
                if isinstance(row['phase_in'], str):
                    phase_in_date = pd.to_datetime(row['phase_in'])
                else:
                    phase_in_date = pd.to_datetime(row['phase_in'])
            except:
                phase_in_date = None
        
        # Create rows for weeks 1 to num_weeks for this product
        for week in range(1, num_weeks + 1):
            # Create a copy of the row and add weeks_since_launch
            row_dict = row.to_dict()
            row_dict['weeks_since_launch'] = week
            
            # Compute ISO week from phase_in date + weeks_since_launch
            if phase_in_date is not None:
                # Calculate the date for this week (phase_in + (week-1) weeks)
                week_date = phase_in_date + timedelta(weeks=week-1)
                # Get ISO week number
                iso_year, iso_week, _ = week_date.isocalendar()
                row_dict['num_week_iso'] = iso_week
            else:
                # If phase_in is not available, set to None (will be handled later)
                row_dict['num_week_iso'] = None
            
            product_rows.append(row_dict)
    
    df_processed = pd.DataFrame(product_rows)
    
    # Create new features from computed num_week_iso
    if 'num_week_iso' in df_processed.columns:
        # Fill missing num_week_iso with 0 (or median if preferred)
        df_processed['num_week_iso'] = df_processed['num_week_iso'].fillna(0).astype(int)
        df_processed["is_week23_peak"] = (df_processed["num_week_iso"] == 23).astype(int)
        # Black Friday is usually ISO week 47-48
        df_processed["is_black_friday_week"] = df_processed["num_week_iso"].isin([47, 48]).astype(int)
    else:
        # If num_week_iso couldn't be computed, set features to 0
        df_processed["is_week23_peak"] = 0
        df_processed["is_black_friday_week"] = 0
    
    # Add historic average demand feature (avg_sales_ID)
    if 'ID' in df_processed.columns:
        if avg_sales_ID_map is not None:
            df_processed["avg_sales_ID"] = df_processed["ID"].map(avg_sales_ID_map)
            # Fill NaN with 0 for products without historical data
            df_processed["avg_sales_ID"] = df_processed["avg_sales_ID"].fillna(0)
        else:
            # If we couldn't load training data, set to 0
            df_processed["avg_sales_ID"] = 0
    
    # Add lag features (set to 0 for test data since we don't have historical sales)
    # These features are needed to match training data structure
    df_processed["lag1"] = 0
    df_processed["lag2"] = 0
    df_processed["lag3"] = 0
    df_processed["lag4"] = 0
    df_processed["lag_mean4"] = 0
    
    # Convert boolean to int if needed
    if 'has_plus_sizes' in df_processed.columns:
        df_processed['has_plus_sizes'] = df_processed['has_plus_sizes'].astype(int)
    
    # Get all features (same as training, excluding target and excluded columns)
    # Note: emb_pca_*, emb_cluster, emb_dist are included as features
    exclude_cols = ['color_name', 'image_embedding', 'embedding_array', 'ID', 'knit_structure', 
                    'num_week_iso', 'weekly_demand', 'Production', 'weekly_sales',
                    'category', 'is_fall', 'waist_type', 'id_season', 'moment', 'woven_structure']
    all_features = [col for col in df_processed.columns if col not in exclude_cols]
    
    # Identify ALL object type columns in test data (CSV loads them as object, not category)
    # Convert them to category dtype to match training data
    object_cols_test = df_processed[all_features].select_dtypes(include=['object']).columns.tolist()
    
    # emb_cluster should be treated as categorical (even though it's numeric)
    # Add it to categorical features if it exists
    if 'emb_cluster' in all_features and 'emb_cluster' not in object_cols_test:
        # Fill any missing values with -1 (sentinel for invalid embeddings)
        if df_processed['emb_cluster'].isna().any():
            df_processed['emb_cluster'] = df_processed['emb_cluster'].fillna(-1)
        # Convert emb_cluster to categorical type
        df_processed['emb_cluster'] = df_processed['emb_cluster'].astype(int).astype('category')
        object_cols_test.append('emb_cluster')
    
    # Also use train_categorical_cols if provided to ensure consistency
    if train_categorical_cols:
        # Combine both lists to ensure we catch all categorical columns
        cols_to_convert = list(set(object_cols_test + [col for col in train_categorical_cols if col in df_processed.columns]))
    else:
        cols_to_convert = object_cols_test
    
    print(f"Converting {len(cols_to_convert)} object columns to categorical in test data")
    
    # Convert object columns to category (matching training data)
    for col in cols_to_convert:
        if col in df_processed.columns:
            # emb_cluster is already handled above, skip it here
            if col == 'emb_cluster':
                continue
            # Fill missing values and convert to category
            df_processed[col] = df_processed[col].astype(str).fillna('MISSING').astype('category')
    
    X_test = df_processed[all_features].copy()
    
    # Keep ID for grouping predictions
    df_meta = df_processed[['ID', 'weeks_since_launch']].copy()
    
    # Log statistics about weeks per product
    weeks_per_product = df_processed.groupby('ID')['weeks_since_launch'].max()
    print(f"Processed {len(df_test)} products into {len(df_processed)} week-product rows")
    print(f"Average weeks per product: {weeks_per_product.mean():.1f}")
    print(f"Min weeks: {weeks_per_product.min()}, Max weeks: {weeks_per_product.max()}")
    
    return X_test, df_meta

def train_lightgbm_model(X_train, y_train, categorical_features, params=None, 
                         num_boost_round=1000, valid_sets=None, verbose_eval=100):
    """Train LightGBM regression model"""
    if params is None:
        params = {
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

def time_based_cv_split(X, y, seasons, train_seasons=[86, 87, 88], val_season=89):
    """
    Split data based on seasons for cross-validation
    Train on train_seasons, validate on val_season
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

def hyperparameter_tuning(X_train, y_train, X_val, y_val, categorical_features, n_trials=100):
    """
    Hyperparameter tuning using Optuna with improved search space
    Includes max_depth tuning and more trials
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

def predict_production(model, X_test, df_meta, penalization_factor=PENALIZATION_FACTOR):
    """
    Predict weekly sales for each week, then sum all weekly_sales per product to get Production.
    
    Steps:
    1. Model predicts weekly_sales for each week-product combination
    2. Apply penalization factor to weekly predictions
    3. Sum all weekly_sales predictions for each product ID to get total Production
    4. Return dataframe with ID and Production columns
    
    Can handle both single model or list of models (ensemble)
    """
    # Handle ensemble (list of models) or single model
    if isinstance(model, list):
        print(f"\nMaking predictions using ensemble of {len(model)} models...")
        # Average predictions across all models
        all_predictions = []
        for i, m in enumerate(model):
            pred = m.predict(X_test)
            all_predictions.append(pred)
        # Average predictions
        weekly_pred = np.mean(all_predictions, axis=0)
    else:
        print(f"\nMaking predictions with single model...")
        weekly_pred = model.predict(X_test)
    
    print(f"Applying {penalization_factor:.1%} penalization factor...")
    
    # Apply penalization factor to avoid being short on demand
    weekly_pred_adjusted = weekly_pred * penalization_factor
    
    # Create prediction dataframe with weekly predictions
    pred_df = df_meta.copy()  # Contains ID and weeks_since_launch
    pred_df['predicted_weekly_sales'] = weekly_pred_adjusted
    
    # Sum all weekly_sales predictions for each product to get total Production
    # Group by ID and sum all weekly predictions
    production_df = pred_df.groupby('ID')['predicted_weekly_sales'].sum().reset_index()
    production_df.columns = ['ID', 'Production']
    
    # Ensure Production is non-negative and integer-like (round to nearest integer)
    production_df['Production'] = np.maximum(0, production_df['Production']).round().astype(int)
    
    return production_df

def get_feature_importance(model, feature_names):
    """Extract and display feature importance"""
    importance = model.feature_importance(importance_type='gain')
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance Ranking:")
    print(feature_imp_df.to_string(index=False))
    
    return feature_imp_df

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Mango Weekly Demand Forecast - Base Model Pipeline")
    print("=" * 60)
    
    # Load training data with seasons
    X, y, categorical_features, seasons = load_and_prepare_train_data(
        'train_processed.csv', return_seasons=True
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
    
    # Step 2: Hyperparameter tuning
    print("\n" + "=" * 60)
    print("Step 2: Hyperparameter Tuning (Improved)")
    print("=" * 60)
    best_params = hyperparameter_tuning(
        X_train_cv, y_train_cv, X_val_cv, y_val_cv, 
        categorical_features, n_trials=100
    )
    
    # Step 3: Train ensemble of models on all training data with best hyperparameters
    print("\n" + "=" * 60)
    print(f"Step 3: Training Ensemble of {N_ENSEMBLE_MODELS} Models on All Training Data")
    print("=" * 60)
    
    models = []
    for i in range(N_ENSEMBLE_MODELS):
        print(f"\nTraining model {i+1}/{N_ENSEMBLE_MODELS}...")
        # Use different random seed for each model in ensemble
        model_params = best_params.copy()
        model_params['random_state'] = 42 + i
        
        model = train_lightgbm_model(
            X, y, categorical_features, 
            params=model_params,
            num_boost_round=1000,
            verbose_eval=100 if i == 0 else 0  # Only verbose for first model
        )
        models.append(model)
    
    # Use first model for feature importance (they should be similar)
    feature_names = list(X.columns)
    feature_imp_df = get_feature_importance(models[0], feature_names)
    
    # Step 4: Process test data and make predictions
    print("\n" + "=" * 60)
    print("Step 4: Making Predictions on Test Data")
    print("=" * 60)
    
    X_test, df_meta = process_test_data('test_processed.csv', max_weeks=30, 
                                        train_categorical_cols=categorical_features)
    
    # Ensure test features match training features (same order and columns)
    # Add any missing columns with default values
    for col in X.columns:
        if col not in X_test.columns:
            print(f"Warning: Missing column {col} in test data, filling with default")
            if col in categorical_features:
                X_test[col] = pd.Categorical(['MISSING'] * len(X_test))
            else:
                X_test[col] = 0
    
    # Reorder test columns to match training
    X_test = X_test[X.columns]
    
    # Ensure categorical columns in test have same categories as training
    for col in categorical_features:
        if col in X_test.columns and col in X.columns:
            # Get training categories
            train_cats = X[col].cat.categories
            # Add any new categories from test that aren't in training
            test_cats = X_test[col].cat.categories
            all_cats = train_cats.union(test_cats)
            # Reorder categories to match training, then add new ones
            X_test[col] = X_test[col].cat.set_categories(all_cats)
    
    # Predict production with penalization using ensemble
    production_df = predict_production(models, X_test, df_meta, PENALIZATION_FACTOR)
    
    # Ensure all test product IDs are in predictions (handle edge cases)
    test_ids = pd.read_csv('test_processed.csv', usecols=['ID'])
    all_test_ids = test_ids['ID'].unique()
    predicted_ids = set(production_df['ID'].unique())
    
    # Add missing IDs with zero production if any
    missing_ids = set(all_test_ids) - predicted_ids
    if missing_ids:
        missing_df = pd.DataFrame({'ID': list(missing_ids), 'Production': 0})
        production_df = pd.concat([production_df, missing_df], ignore_index=True)
    
    # Sort by ID to match submission format
    production_df = production_df.sort_values('ID').reset_index(drop=True)
    
    # Save production predictions
    output_file = 'predictions.csv'
    production_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    print(f"Total products: {len(production_df)}")
    print(f"Total predicted production: {production_df['Production'].sum():,}")
    print(f"Average production per product: {production_df['Production'].mean():.2f}")
    print(f"Min production: {production_df['Production'].min()}")
    print(f"Max production: {production_df['Production'].max()}")
    
    # Save feature importance
    feature_imp_df.to_csv('feature_importance.csv', index=False)
    print(f"\nFeature importance saved to feature_importance.csv")
    
    print("\n" + "=" * 60)
    print("Training and prediction completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()


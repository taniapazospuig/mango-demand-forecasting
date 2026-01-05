"""Data preprocessing functions for loading and preparing train/test data."""

import pandas as pd
import numpy as np


def load_and_prepare_train_data(filepath, return_seasons=False):
    """
    Load and prepare training data from processed CSV.
    
    Args:
        filepath: Path to processed training CSV
        return_seasons: If True, return seasons array
        
    Returns:
        X: Feature dataframe
        y: Target array (weekly_demand)
        categorical_features: List of categorical feature names
        seasons: (optional) Array of season IDs
        bin_info: Dict with binning information for num_stores_bin and num_sizes_bin
    """
    df = pd.read_csv(filepath)
    
    # Target variable
    target_col = 'weekly_demand'
    if target_col not in df.columns:
        # Fallback to weekly_sales if weekly_demand doesn't exist
        target_col = 'weekly_sales'
    
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    
    # Remove metadata columns that shouldn't be features
    metadata_cols = ['ID', 'id_season', 'year', 'num_week_iso', 'weekly_sales']
    if 'weekly_demand' in X.columns:
        metadata_cols.append('weekly_demand')
    
    # Keep only feature columns
    feature_cols = [col for col in X.columns if col not in metadata_cols]
    X = X[feature_cols].copy()
    
    # Identify categorical features
    # These are typically: object dtype, bin columns, cluster columns, or explicitly categorical
    categorical_features = []
    for col in X.columns:
        is_categorical = (
            X[col].dtype == 'object' or 
            X[col].dtype.name == 'category' or
            col.endswith('_bin') or 
            'cluster' in col.lower() or
            col in ['family', 'color_name', 'print_type', 'waist_type', 'length_type',
                   'silhouette_type', 'neck_lapel_type', 'sleeve_length_type',
                   'woven_structure', 'knit_structure']
        )
        if is_categorical:
            categorical_features.append(col)
            # Convert to categorical if not already
            if X[col].dtype != 'category':
                X[col] = X[col].astype('category')
    
    # Extract bin_info for num_stores_bin and num_sizes_bin
    bin_info = {}
    for bin_col in ['num_stores_bin', 'num_sizes_bin']:
        if bin_col in X.columns:
            bin_info[bin_col] = {
                'categories': X[bin_col].cat.categories.tolist()
            }
    
    # Extract seasons if requested
    seasons = None
    if return_seasons:
        if 'id_season' in df.columns:
            seasons = df['id_season'].values
        else:
            # If no season column, create dummy seasons
            seasons = np.zeros(len(df), dtype=int)
    
    if return_seasons:
        return X, y, categorical_features, seasons, bin_info
    else:
        return X, y, categorical_features, bin_info


def process_test_data(filepath, max_weeks=30, train_categorical_cols=None, bin_info=None):
    """
    Load and prepare test data from processed CSV.
    
    Args:
        filepath: Path to processed test CSV
        max_weeks: Maximum number of weeks per product to include
        train_categorical_cols: List of categorical columns from training data
        bin_info: Dict with binning information from training data
        
    Returns:
        X_test: Feature dataframe
        df_meta: Metadata dataframe with ID and other info
    """
    df_test = pd.read_csv(filepath)
    
    # Limit to max_weeks per product
    if 'weeks_since_launch' in df_test.columns:
        df_test = df_test[df_test['weeks_since_launch'] < max_weeks].copy()
    
    # Separate metadata
    metadata_cols = ['ID']
    if 'id_season' in df_test.columns:
        metadata_cols.append('id_season')
    if 'year' in df_test.columns:
        metadata_cols.append('year')
    if 'num_week_iso' in df_test.columns:
        metadata_cols.append('num_week_iso')
    if 'weeks_since_launch' in df_test.columns:
        metadata_cols.append('weeks_since_launch')
    
    df_meta = df_test[metadata_cols].copy()
    
    # Remove metadata and target columns (if any)
    cols_to_drop = metadata_cols + ['weekly_sales', 'weekly_demand']
    feature_cols = [col for col in df_test.columns if col not in cols_to_drop]
    X_test = df_test[feature_cols].copy()
    
    # Handle categorical features
    if train_categorical_cols:
        for col in train_categorical_cols:
            if col in X_test.columns:
                if X_test[col].dtype != 'category':
                    X_test[col] = X_test[col].astype('category')
            elif col in ['num_stores_bin', 'num_sizes_bin'] and bin_info and col in bin_info:
                # Create missing bin columns with default category
                default_cat = bin_info[col]['categories'][0] if bin_info[col]['categories'] else '0'
                X_test[col] = pd.Categorical([default_cat] * len(X_test), categories=bin_info[col]['categories'])
            else:
                # Create missing categorical column
                X_test[col] = pd.Categorical(['MISSING'] * len(X_test))
    
    return X_test, df_meta


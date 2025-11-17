"""Data preprocessing functions for loading and preparing training/test data."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from ..features.engineering import bin_numeric_features, apply_binning_with_info
from ..utils.config import EXCLUDED_COLUMNS, LOW_IMPORTANCE_FEATURES


def load_and_prepare_train_data(
    filepath: str = 'train_processed.csv',
    return_seasons: bool = False
) -> Tuple:
    """
    Load training data and select required features.
    
    Parameters:
    -----------
    filepath : str
        Path to the processed training CSV file
    return_seasons : bool
        Whether to return season information for time-based CV
        
    Returns:
    --------
    Tuple containing:
        - X: Feature matrix
        - y: Target variable
        - categorical_features: List of categorical feature names
        - seasons (optional): Season IDs for time-based CV
        - bin_info: Binning information for consistency with test data
    """
    print("Loading training data...")
    df = pd.read_csv(filepath)
    
    # Apply binning to num_stores and num_sizes to prevent overfitting
    print("Binning numeric features (num_stores, num_sizes)...")
    df, bin_info = bin_numeric_features(df)
    
    # Build exclusion list
    exclude_cols = EXCLUDED_COLUMNS.copy()
    
    # Exclude all PCA embedding features
    pca_features = [col for col in df.columns if col.startswith('emb_pca_')]
    exclude_cols.extend(pca_features)
    
    # Exclude low importance features
    for feat in LOW_IMPORTANCE_FEATURES:
        if feat in df.columns and feat not in exclude_cols:
            exclude_cols.append(feat)
    
    # Get all features
    all_features = [col for col in df.columns if col not in exclude_cols]
    
    # Identify categorical features
    object_cols = df[all_features].select_dtypes(include=['object']).columns.tolist()
    
    # Add binned features to categorical
    for bin_feat in ['num_stores_bin', 'num_sizes_bin']:
        if bin_feat in all_features and bin_feat not in object_cols:
            object_cols.append(bin_feat)
    
    # Handle emb_cluster and color_cluster as categorical
    for cluster_feat in ['emb_cluster', 'color_cluster']:
        if cluster_feat in all_features and cluster_feat not in object_cols:
            if df[cluster_feat].isna().any():
                df[cluster_feat] = df[cluster_feat].fillna(-1)
            df[cluster_feat] = df[cluster_feat].astype(int).astype('category')
            object_cols.append(cluster_feat)
    
    # Count feature types
    cluster_features = [col for col in all_features if col.startswith('cluster_')]
    similar_features = [col for col in all_features if col.startswith('similar_product_')]
    family_features = [col for col in all_features if col.startswith('family_')]
    
    print(f"Found {len(all_features)} features, {len(object_cols)} categorical columns")
    print(f"  Excluded: waist_type, is_fall, category, id_season, moment, woven_structure, all PCA components")
    
    # Print key features
    key_features = {
        'velocity_1_3': 'velocity_1_3',
        'trend_score': ['sim_to_top', 'sim_to_bottom', 'trend_score'],
        'emb_cluster': 'emb_cluster',
        'emb_dist': 'emb_dist',
        'color_cluster': 'color_cluster',
        'color_cluster_dist': 'color_cluster_dist',
        'is_week_23': 'is_week_23',
        'is_black_friday': 'is_black_friday',
        'num_stores_bin': 'num_stores_bin',
        'num_sizes_bin': 'num_sizes_bin',
    }
    
    for name, feat in key_features.items():
        if isinstance(feat, list):
            if any(f in all_features for f in feat):
                print(f"  ✓ trend_score features ({', '.join(feat)})")
        elif feat in all_features:
            print(f"  ✓ {name}")
    
    if cluster_features:
        print(f"  ✓ {len(cluster_features)} cluster-level features")
    if similar_features:
        print(f"  ✓ {len(similar_features)} similarity features")
    if family_features:
        print(f"  ✓ {len(family_features)} family-level features")
    
    # Convert object columns to category dtype
    for col in object_cols:
        if col in df.columns:
            if col in ['emb_cluster', 'color_cluster']:
                continue
            if col in ['num_stores_bin', 'num_sizes_bin']:
                if df[col].dtype.name != 'category':
                    df[col] = df[col].astype('category')
                continue
            df[col] = df[col].astype(str).fillna('MISSING').astype('category')
    
    # Create feature matrix and target
    X = df[all_features].copy()
    y = df['weekly_sales'].copy()
    
    if return_seasons:
        seasons = df['id_season'].copy() if 'id_season' in df.columns else None
        return X, y, object_cols, seasons, bin_info
    else:
        return X, y, object_cols, bin_info


def process_test_data(
    filepath: str = 'test_processed.csv',
    max_weeks: Optional[int] = None,
    train_categorical_cols: Optional[List[str]] = None,
    bin_info: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process preprocessed test data to create rows for multiple weeks per product.
    
    Parameters:
    -----------
    filepath : str
        Path to the processed test CSV file
    max_weeks : int, optional
        Maximum number of weeks to predict per product (default: 30)
    train_categorical_cols : list, optional
        List of categorical columns from training data for consistency
    bin_info : dict, optional
        Bin information from training data to ensure consistent binning
        
    Returns:
    --------
    Tuple containing:
        - X_test: Feature matrix for test data
        - df_meta: Metadata dataframe with ID and weeks_since_launch
    """
    print("Loading preprocessed test data...")
    df_test = pd.read_csv(filepath)
    
    if max_weeks is None:
        max_weeks = 30
    
    # Create weeks_since_launch for each product-week combination
    product_rows = []
    for _, row in df_test.iterrows():
        if 'life_cycle_length' in row and pd.notna(row['life_cycle_length']):
            num_weeks = int(row['life_cycle_length'])
            num_weeks = max(1, min(num_weeks, max_weeks))
        else:
            num_weeks = max_weeks
        
        for week in range(1, num_weeks + 1):
            row_dict = row.to_dict()
            row_dict['weeks_since_launch'] = week
            product_rows.append(row_dict)
    
    df_processed = pd.DataFrame(product_rows)
    
    # Convert boolean to int if needed
    if 'has_plus_sizes' in df_processed.columns:
        df_processed['has_plus_sizes'] = df_processed['has_plus_sizes'].astype(int)
    
    # Apply binning
    if bin_info:
        print("Applying binning to test data using training bin definitions...")
        df_processed = apply_binning_with_info(df_processed, bin_info)
    else:
        print("Warning: No bin_info provided, creating bins from test data...")
        from ..features.engineering import bin_numeric_features
        df_processed, _ = bin_numeric_features(df_processed)
    
    # Add seasonality features
    if 'num_week_iso' in df_processed.columns:
        df_processed['is_week_23'] = (df_processed['num_week_iso'] == 23).astype(int)
        df_processed['is_black_friday'] = (df_processed['num_week_iso'].isin([47, 48])).astype(int)
    else:
        df_processed['is_week_23'] = 0
        df_processed['is_black_friday'] = 0
        print("Note: num_week_iso not in test data - seasonality features set to 0")
    
    # Build exclusion list (same as training)
    exclude_cols = EXCLUDED_COLUMNS.copy()
    pca_features = [col for col in df_processed.columns if col.startswith('emb_pca_')]
    exclude_cols.extend(pca_features)
    
    for feat in LOW_IMPORTANCE_FEATURES:
        if feat in df_processed.columns and feat not in exclude_cols:
            exclude_cols.append(feat)
    
    all_features = [col for col in df_processed.columns if col not in exclude_cols]
    
    # Identify categorical columns
    object_cols_test = df_processed[all_features].select_dtypes(include=['object']).columns.tolist()
    
    # Add binned features
    for bin_feat in ['num_stores_bin', 'num_sizes_bin']:
        if bin_feat in all_features and bin_feat not in object_cols_test:
            object_cols_test.append(bin_feat)
    
    # Handle cluster features
    for cluster_feat in ['emb_cluster', 'color_cluster']:
        if cluster_feat in all_features and cluster_feat not in object_cols_test:
            if df_processed[cluster_feat].isna().any():
                df_processed[cluster_feat] = df_processed[cluster_feat].fillna(-1)
            df_processed[cluster_feat] = df_processed[cluster_feat].astype(int).astype('category')
            object_cols_test.append(cluster_feat)
    
    # Combine with training categorical columns
    if train_categorical_cols:
        cols_to_convert = list(set(object_cols_test + [
            col for col in train_categorical_cols if col in df_processed.columns
        ]))
    else:
        cols_to_convert = object_cols_test
    
    print(f"Converting {len(cols_to_convert)} object columns to categorical in test data")
    
    # Convert to category
    for col in cols_to_convert:
        if col in df_processed.columns:
            if col in ['emb_cluster', 'color_cluster']:
                continue
            if col in ['num_stores_bin', 'num_sizes_bin']:
                if df_processed[col].dtype.name != 'category':
                    df_processed[col] = df_processed[col].astype('category')
                continue
            df_processed[col] = df_processed[col].astype(str).fillna('MISSING').astype('category')
    
    X_test = df_processed[all_features].copy()
    df_meta = df_processed[['ID', 'weeks_since_launch']].copy()
    
    # Log statistics
    weeks_per_product = df_processed.groupby('ID')['weeks_since_launch'].max()
    print(f"Processed {len(df_test)} products into {len(df_processed)} week-product rows")
    print(f"Average weeks per product: {weeks_per_product.mean():.1f}")
    print(f"Min weeks: {weeks_per_product.min()}, Max weeks: {weeks_per_product.max()}")
    
    return X_test, df_meta


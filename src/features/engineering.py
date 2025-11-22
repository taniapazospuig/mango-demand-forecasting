"""Feature engineering functions including binning and transformations."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def bin_numeric_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Bin num_stores and num_sizes to prevent overfitting to exact numeric values.
    
    num_stores bins: 0-100, 100-200, 200-300, ..., 1100-inf (12 bins total)
    num_sizes bins: 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-10, 10-inf (10 bins total)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    Tuple containing:
        - df: Dataframe with binned features
        - bin_info: Dictionary with bin information for consistency
    """
    df = df.copy()
    bin_info = {}
    
    # Bin num_stores: 12 bins (0-100, 100-200, ..., 1100-inf)
    if 'num_stores' in df.columns:
        bins_stores = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, float('inf')]
        labels_stores = ['0-100', '100-200', '200-300', '300-400', '400-500', '500-600',
                        '600-700', '700-800', '800-900', '900-1000', '1000-1100', '1100+']
        
        df['num_stores'] = df['num_stores'].fillna(0)
        df['num_stores_bin'] = pd.cut(
            df['num_stores'], bins=bins_stores, labels=labels_stores,
            right=False, include_lowest=True
        )
        df['num_stores_bin'] = df['num_stores_bin'].cat.add_categories(['UNKNOWN']).fillna('UNKNOWN')
        
        bin_info['num_stores'] = {'bins': bins_stores, 'labels': labels_stores}
        print(f"  ✓ Binned num_stores into {len(labels_stores)} categories")
    
    # Bin num_sizes: 10 bins (1-2, 2-3, ..., 9-10, 10+)
    if 'num_sizes' in df.columns:
        bins_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
        labels_sizes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']
        
        df['num_sizes'] = df['num_sizes'].fillna(1)
        df['num_sizes'] = df['num_sizes'].clip(lower=1)
        df['num_sizes_bin'] = pd.cut(
            df['num_sizes'], bins=bins_sizes, labels=labels_sizes,
            right=False, include_lowest=True
        )
        df['num_sizes_bin'] = df['num_sizes_bin'].cat.add_categories(['UNKNOWN']).fillna('UNKNOWN')
        
        bin_info['num_sizes'] = {'bins': bins_sizes, 'labels': labels_sizes}
        print(f"  ✓ Binned num_sizes into {len(labels_sizes)} categories")
    
    return df, bin_info


def apply_binning_with_info(df: pd.DataFrame, bin_info: Dict) -> pd.DataFrame:
    """
    Apply binning to test data using the same bin definitions from training data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Test dataframe
    bin_info : dict
        Bin information from training data
        
    Returns:
    --------
    pd.DataFrame with binned features applied
    """
    df = df.copy()
    
    # Apply num_stores binning
    if 'num_stores' in bin_info and 'num_stores' in df.columns:
        bins_stores = bin_info['num_stores']['bins']
        labels_stores = bin_info['num_stores']['labels']
        
        df['num_stores'] = df['num_stores'].fillna(0)
        df['num_stores_bin'] = pd.cut(
            df['num_stores'], bins=bins_stores, labels=labels_stores,
            right=False, include_lowest=True
        )
        df['num_stores_bin'] = df['num_stores_bin'].cat.add_categories(['UNKNOWN']).fillna('UNKNOWN')
    
    # Apply num_sizes binning
    if 'num_sizes' in bin_info and 'num_sizes' in df.columns:
        bins_sizes = bin_info['num_sizes']['bins']
        labels_sizes = bin_info['num_sizes']['labels']
        
        df['num_sizes'] = df['num_sizes'].fillna(1)
        df['num_sizes'] = df['num_sizes'].clip(lower=1)
        df['num_sizes_bin'] = pd.cut(
            df['num_sizes'], bins=bins_sizes, labels=labels_sizes,
            right=False, include_lowest=True
        )
        df['num_sizes_bin'] = df['num_sizes_bin'].cat.add_categories(['UNKNOWN']).fillna('UNKNOWN')
    
    return df


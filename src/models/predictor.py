"""Prediction functions for generating production forecasts."""

import pandas as pd
import numpy as np
from typing import Union, List
import lightgbm as lgb
from ..utils.config import PENALIZATION_FACTOR


def predict_production(
    model: Union[lgb.Booster, List[lgb.Booster]],
    X_test: pd.DataFrame,
    df_meta: pd.DataFrame,
    penalization_factor: float = PENALIZATION_FACTOR,
    save_weekly_debug: bool = False
) -> pd.DataFrame:
    """
    Predict weekly sales for each week, then sum all weekly_sales per product to get Production.
    
    Steps:
    1. Model predicts weekly_sales for each week-product combination
    2. Apply penalization factor to weekly predictions
    3. Sum all weekly_sales predictions for each product ID to get total Production
    4. Return dataframe with ID and Production columns
    
    Parameters:
    -----------
    model : lgb.Booster or list
        Single model or list of models (ensemble)
    X_test : pd.DataFrame
        Test feature matrix
    df_meta : pd.DataFrame
        Metadata dataframe with ID and weeks_since_launch
    penalization_factor : float
        Factor to multiply predictions by (default: PENALIZATION_FACTOR)
    save_weekly_debug : bool
        If True, saves weekly predictions breakdown to CSV
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with ID and Production columns
    """
    # Handle ensemble (list of models) or single model
    if isinstance(model, list):
        print(f"\nMaking predictions using ensemble of {len(model)} models...")
        all_predictions = []
        for i, m in enumerate(model):
            pred = m.predict(X_test)
            all_predictions.append(pred)
        weekly_pred = np.mean(all_predictions, axis=0)
    else:
        print(f"\nMaking predictions with single model...")
        weekly_pred = model.predict(X_test)
    
    print(f"Applying {penalization_factor:.1%} penalization factor...")
    
    # Apply penalization factor to avoid being short on demand
    weekly_pred_adjusted = weekly_pred * penalization_factor
    
    # Create prediction dataframe with weekly predictions
    pred_df = df_meta.copy()
    pred_df['predicted_weekly_sales'] = weekly_pred_adjusted
    pred_df['predicted_weekly_sales_raw'] = weekly_pred
    
    # Sum all weekly_sales predictions for each product to get total Production
    production_df = pred_df.groupby('ID')['predicted_weekly_sales'].sum().reset_index()
    production_df.columns = ['ID', 'Production']
    
    # Debug: Check for negative or very low predictions
    zero_or_negative = production_df['Production'] <= 0
    if zero_or_negative.sum() > 0:
        print(f"\nWarning: {zero_or_negative.sum()} products have Production <= 0 before rounding")
        print("Sample products with negative/zero predictions:")
        sample_ids = production_df[zero_or_negative]['ID'].head(10).tolist()
        sample_preds = pred_df[pred_df['ID'].isin(sample_ids)].groupby('ID')['predicted_weekly_sales'].agg(['sum', 'mean', 'min', 'max'])
        print(sample_preds)
    
    # Ensure Production is non-negative and integer-like
    production_before_round = np.maximum(0, production_df['Production'])
    
    # Check if products have any positive weekly predictions
    products_with_positive_weeks = pred_df[pred_df['predicted_weekly_sales'] > 0]['ID'].unique()
    
    # Apply rounding
    production_df['Production'] = production_before_round.round().astype(int)
    
    # Set minimum of 1 for products that had any positive weekly prediction but rounded to 0
    has_positive_weeks = production_df['ID'].isin(products_with_positive_weeks)
    was_zero = production_df['Production'] == 0
    fix_mask = has_positive_weeks & was_zero
    if fix_mask.sum() > 0:
        print(f"\nFixing {fix_mask.sum()} products: Had positive weekly predictions but Production rounded to 0")
        print("Setting minimum Production = 1 for these products")
        production_df.loc[fix_mask, 'Production'] = 1
    
    # Additional check: products with 0 production
    zero_count = (production_df['Production'] == 0).sum()
    if zero_count > 0:
        print(f"\nAfter rounding and fixes: {zero_count} products still have Production = 0")
        print("These products had no positive weekly sales predictions")
    
    # Save weekly predictions for debugging/root cause analysis
    if save_weekly_debug:
        import os
        os.makedirs('outputs', exist_ok=True)
        debug_file = 'outputs/weekly_predictions_debug.csv'
        debug_cols = ['ID', 'weeks_since_launch', 'predicted_weekly_sales', 'predicted_weekly_sales_raw']
        pred_df[debug_cols].to_csv(debug_file, index=False)
        print(f"\nSaved weekly predictions breakdown to {debug_file} for root cause analysis")
    
    return production_df


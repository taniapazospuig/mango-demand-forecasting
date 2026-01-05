"""Prediction functions for generating production forecasts."""

import pandas as pd
import numpy as np


def predict_production(models, X_test, df_meta, penalization_factor=1.0, save_weekly_debug=False):
    """
    Predict production quantities for test products using ensemble of models.
    
    Args:
        models: List of trained LightGBM models
        X_test: Test feature dataframe
        df_meta: Metadata dataframe with ID and other info
        penalization_factor: Factor to multiply predictions by (to avoid stockouts)
        save_weekly_debug: If True, save weekly predictions
        
    Returns:
        DataFrame with columns (ID, Production)
    """
    # Make predictions with ensemble
    predictions = []
    for model in models:
        pred = model.predict(X_test.values)
        predictions.append(pred)
    
    # Average predictions across models
    y_pred = np.mean(predictions, axis=0)
    
    # Apply penalization factor
    y_pred = y_pred * penalization_factor
    
    # Create predictions dataframe
    pred_df = df_meta.copy()
    pred_df['predicted_demand'] = y_pred
    
    # Aggregate to total production per product
    # Sum weekly predictions for each product
    production_df = pred_df.groupby('ID')['predicted_demand'].sum().reset_index()
    production_df.columns = ['ID', 'Production']
    
    # Ensure Production is non-negative and integer
    production_df['Production'] = production_df['Production'].clip(lower=0)
    production_df['Production'] = production_df['Production'].round().astype(int)
    
    if save_weekly_debug:
        # Save weekly predictions for debugging
        pred_df.to_csv('outputs/weekly_predictions_debug.csv', index=False)
        print("Weekly predictions saved to outputs/weekly_predictions_debug.csv")
    
    return production_df


"""
Mango Weekly Demand Forecast Model - Base Pipeline
LightGBM regression model to forecast weekly sales for fashion products
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
PENALIZATION_FACTOR = 1.05  # Increase predictions by 5% to avoid being short on demand

def load_and_prepare_train_data(filepath='train_processed.csv'):
    """Load training data and select required features"""
    print("Loading training data...")
    df = pd.read_csv(filepath)
    
    # Select features as specified
    features = ['weeks_since_launch', 'R', 'G', 'B', 'num_stores', 
                'num_sizes', 'price', 'is_fall']
    target = 'weekly_sales'
    
    # Verify all features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Create feature matrix and target
    X = df[features].copy()
    y = df[target].copy()
    
    return X, y

def process_test_data(filepath='data/test.csv', max_weeks=30):
    """
    Process test.csv to extract features needed for prediction
    Creates rows for multiple weeks per product (weeks_since_launch = 1 to max_weeks)
    """
    print("Loading and processing test data...")
    df_test = pd.read_csv(filepath, sep=';')
    
    # Parse RGB from color_rgb column (format: "245,244,238")
    def parse_rgb(x):
        parts = str(x).split(",")
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    
    df_test[["R", "G", "B"]] = df_test["color_rgb"].apply(lambda x: pd.Series(parse_rgb(x)))
    
    # Normalize RGB values to 0-1 range (divide by 255)
    df_test[["R", "G", "B"]] = df_test[["R", "G", "B"]] / 255.0
    
    # Create is_fall boolean column based on id_season
    # Pattern: even id_season values are fall (1), odd values are not fall (0)
    df_test['is_fall'] = (df_test['id_season'] % 2 == 0).astype(int)
    
    # Create rows for multiple weeks per product
    # For each product, create rows for weeks_since_launch = 1 to max_weeks
    product_rows = []
    for _, row in df_test.iterrows():
        for week in range(1, max_weeks + 1):
            product_rows.append({
                'ID': row['ID'],
                'weeks_since_launch': week,
                'R': row['R'],
                'G': row['G'],
                'B': row['B'],
                'num_stores': row['num_stores'],
                'num_sizes': row['num_sizes'],
                'price': row['price'],
                'is_fall': row['is_fall']
            })
    
    df_processed = pd.DataFrame(product_rows)
    
    # Select features
    features = ['weeks_since_launch', 'R', 'G', 'B', 'num_stores', 
                'num_sizes', 'price', 'is_fall']
    X_test = df_processed[features].copy()
    
    # Keep ID for grouping predictions
    df_meta = df_processed[['ID', 'weeks_since_launch']].copy()
    
    print(f"Processed {len(df_test)} products into {len(df_processed)} week-product rows")
    
    return X_test, df_meta

def train_lightgbm_model(X_train, y_train):
    """Train LightGBM regression model on all data"""
    print("\nTraining LightGBM model on all data...")
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Parameters
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
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=[lgb.log_evaluation(period=100)]
    )
    
    return model

def predict_production(model, X_test, df_meta, penalization_factor=PENALIZATION_FACTOR):
    """
    Predict weekly sales, apply penalization factor, and aggregate to production per product
    """
    print(f"\nMaking predictions (with {penalization_factor:.1%} penalization factor)...")
    
    # Predict weekly sales for each week-product combination
    weekly_pred = model.predict(X_test)
    
    # Apply penalization factor to avoid being short on demand
    weekly_pred_adjusted = weekly_pred * penalization_factor
    
    # Create prediction dataframe
    pred_df = df_meta.copy()
    pred_df['predicted_weekly_sales'] = weekly_pred_adjusted
    
    # Aggregate by product to get total production (sum of all weeks)
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
    
    # Load training data
    X_train, y_train = load_and_prepare_train_data('train_processed.csv')
    print(f"Loaded {len(X_train):,} training samples")
    
    # Train model on all training data
    model = train_lightgbm_model(X_train, y_train)
    
    # Feature importance
    feature_names = ['weeks_since_launch', 'R', 'G', 'B', 'num_stores', 
                     'num_sizes', 'price', 'is_fall']
    feature_imp_df = get_feature_importance(model, feature_names)
    
    # Process test data and make predictions
    X_test, df_meta = process_test_data('data/test.csv', max_weeks=30)
    
    # Predict production with penalization
    production_df = predict_production(model, X_test, df_meta, PENALIZATION_FACTOR)
    
    # Ensure all test product IDs are in predictions (handle edge cases)
    test_ids = pd.read_csv('data/test.csv', sep=';', usecols=['ID'])
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


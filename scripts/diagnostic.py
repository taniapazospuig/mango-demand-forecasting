"""
Diagnostic script to investigate why products get 0 or 1 predictions
Uses the actual predictions.csv file - no need to re-predict

WHAT THIS SCRIPT DOES:
=====================
This diagnostic tool helps you understand why certain products received very low 
predictions (0 or 1) despite having features that suggest they should perform well.

The script performs the following analysis for each problematic product:

1. FEATURE ANALYSIS:
   - Examines key predictive features (price, num_stores, num_sizes, etc.)
   - Compares feature values with training data distribution
   - Flags features that are out of training range (could confuse the model)

2. CATEGORICAL FEATURE CHECK:
   - Verifies if categorical values (family, emb_cluster, color_cluster, etc.) 
    were seen during training
   - Flags new categories that the model hasn't learned from

3. MISSING/DEFAULT VALUES:
   - Identifies features with missing, zero, or default values
   - These might indicate data quality issues or incorrect feature engineering

4. SIMILAR PRODUCT COMPARISON:
   - Finds products in training data with similar characteristics
   - Compares their actual sales with the problematic product's prediction
   - Helps identify if the prediction is unreasonably low

5. ROOT CAUSE IDENTIFICATION:
   - Identifies likely causes: negative predictions, very small values, 
    short life cycles, feature mismatches, etc.

6. RECOMMENDATIONS:
   - Provides actionable suggestions to fix the issue

USAGE:
======
# Analyze specific products
python diagnostic.py --ids 224 1691 3210

# Analyze all products with Production = 0 or 1
python diagnostic.py --all-problematic

# Analyze top N problematic products
python diagnostic.py --top-n 10
"""

import pandas as pd
import numpy as np
import argparse
import sys

def diagnose_zero_predictions(predictions_path='outputs/predictions.csv',
                              test_processed_path='data/processed/test_processed.csv', 
                              train_processed_path='data/processed/train_processed.csv',
                              problem_ids=None,
                              analyze_all_problematic=False,
                              top_n=None):
    """
    Diagnose why specific products get 0 or 1 predictions
    
    Parameters:
    -----------
    predictions_path : str
        Path to predictions.csv file
    test_processed_path : str
        Path to test_processed.csv file
    train_processed_path : str
        Path to train_processed.csv file
    problem_ids : list, optional
        Specific product IDs to analyze. If None, will analyze all problematic products
    analyze_all_problematic : bool
        If True, analyze all products with Production = 0 or 1
    top_n : int, optional
        Analyze top N problematic products (by ID or by some criteria)
    """
    print("=" * 80)
    print("DIAGNOSING ZERO/LOW PREDICTIONS")
    print("=" * 80)
    print("\nLoading data files...")
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_path)
    print(f"✓ Loaded {len(predictions_df)} predictions from {predictions_path}")
    
    # Load test and train data
    test_df = pd.read_csv(test_processed_path)
    print(f"✓ Loaded {len(test_df)} test products from {test_processed_path}")
    
    train_df = pd.read_csv(train_processed_path)
    print(f"✓ Loaded {len(train_df)} training rows from {train_processed_path}")
    
    # Identify problematic products
    zero_products = predictions_df[predictions_df['Production'] == 0]['ID'].tolist()
    one_products = predictions_df[predictions_df['Production'] == 1]['ID'].tolist()
    problematic_ids = zero_products + one_products
    
    print(f"\n📊 PROBLEMATIC PRODUCTS SUMMARY:")
    print(f"   Products with Production = 0: {len(zero_products)}")
    print(f"   Products with Production = 1: {len(one_products)}")
    print(f"   Total problematic: {len(problematic_ids)}")
    
    # Determine which products to analyze
    if problem_ids is None:
        if analyze_all_problematic:
            problem_ids = problematic_ids
            print(f"\n🔍 Analyzing ALL {len(problematic_ids)} problematic products...")
        elif top_n is not None:
            problem_ids = problematic_ids[:top_n]
            print(f"\n🔍 Analyzing top {len(problem_ids)} problematic products...")
        else:
            # Default: analyze first 5 problematic products
            problem_ids = problematic_ids[:5]
            print(f"\n🔍 Analyzing first {len(problem_ids)} problematic products (use --all-problematic for all)...")
    else:
        print(f"\n🔍 Analyzing {len(problem_ids)} specified products...")
    
    # Analyze each problem product
    for idx, product_id in enumerate(problem_ids, 1):
        print(f"\n{'='*80}")
        print(f"ANALYZING PRODUCT {idx}/{len(problem_ids)}: ID = {product_id}")
        print(f"{'='*80}")
        
        # Get product's prediction
        product_pred = predictions_df[predictions_df['ID'] == product_id]
        if len(product_pred) == 0:
            print(f"❌ ERROR: Product {product_id} not found in predictions.csv!")
            continue
        
        production_value = product_pred['Production'].iloc[0]
        print(f"\n📦 Production Prediction: {production_value}")
        
        # Get product's features from test data
        product_features = test_df[test_df['ID'] == product_id]
        if len(product_features) == 0:
            print(f"❌ ERROR: Product {product_id} not found in test_processed.csv!")
            continue
        
        product_features = product_features.iloc[0]  # Get first row (all rows have same features)
        
        # ========== 1. KEY FEATURE ANALYSIS ==========
        print(f"\n{'─'*80}")
        print("1️⃣  KEY FEATURE ANALYSIS")
        print(f"{'─'*80}")
        print("   Comparing product features with training data distribution...")
        
        key_features = ['price', 'num_stores', 'num_sizes', 'life_cycle_length', 
                       'phase_in', 'phase_out', 'velocity_1_3', 'trend_score',
                       'sim_to_top', 'sim_to_bottom', 'emb_dist', 'color_cluster_dist',
                       'cluster_demand_last_season', 'similar_product_demand_mean',
                       'family_velocity_1_3_last_season', 'family_demand_mean_last_season']
        
        feature_issues = []
        for feat in key_features:
            if feat in product_features.index:
                value = product_features[feat]
                
                # Skip if value is NaN
                if pd.isna(value):
                    print(f"   ⚠️  {feat:35s}: NaN (MISSING)")
                    feature_issues.append((feat, 'missing'))
                    continue
                
                # Compare with training distribution
                if feat in train_df.columns:
                    train_values = train_df[feat].dropna()
                    
                    if len(train_values) > 0:
                        train_min = train_values.min()
                        train_max = train_values.max()
                        train_mean = train_values.mean()
                        train_median = train_values.median()
                        train_q25 = train_values.quantile(0.25)
                        train_q75 = train_values.quantile(0.75)
                        
                        # Format value based on type
                        if isinstance(value, (int, float)):
                            value_str = f"{value:15.2f}"
                        else:
                            value_str = str(value)
                        
                        # Determine if value is problematic
                        is_outlier = False
                        issue_msg = ""
                        
                        if value < train_min or value > train_max:
                            is_outlier = True
                            issue_msg = " ⚠️  OUT OF TRAINING RANGE!"
                            feature_issues.append((feat, 'out_of_range'))
                        elif value < train_q25:
                            issue_msg = " (below 25th percentile)"
                        elif value > train_q75:
                            issue_msg = " (above 75th percentile)"
                        
                        # Show comparison
                        print(f"   {feat:35s}: {value_str} | Train: [{train_min:.2f}, {train_max:.2f}], "
                              f"median={train_median:.2f}{issue_msg}")
                        
                        # Special checks for important features
                        if feat == 'price' and value > train_q75:
                            print(f"      ⚠️  High price might reduce demand")
                        if feat == 'num_stores' and value < train_q25:
                            print(f"      ⚠️  Low num_stores might limit sales")
                        if feat == 'velocity_1_3' and value < train_q25:
                            print(f"      ⚠️  Low velocity_1_3 suggests slow-selling cluster")
                    else:
                        print(f"   {feat:35s}: {value} | No training data available")
                else:
                    print(f"   {feat:35s}: {value} | Feature not in training data")
        
        # ========== 2. CATEGORICAL FEATURE CHECK ==========
        print(f"\n{'─'*80}")
        print("2️⃣  CATEGORICAL FEATURE CHECK")
        print(f"{'─'*80}")
        print("   Verifying if categorical values were seen during training...")
        
        cat_features = ['family', 'emb_cluster', 'color_cluster', 'silhouette_type', 
                       'neck_lapel_type', 'sleeve_length_type', 'fabric', 'print_type']
        
        categorical_issues = []
        for feat in cat_features:
            if feat in product_features.index:
                value = product_features[feat]
                
                if pd.isna(value):
                    print(f"   ⚠️  {feat:35s}: NaN (MISSING)")
                    categorical_issues.append((feat, 'missing'))
                    continue
                
                # Check if this category exists in training
                if feat in train_df.columns:
                    train_categories = set(train_df[feat].dropna().unique())
                    
                    if value not in train_categories:
                        print(f"   ⚠️  {feat:35s}: {value} ⚠️  NOT SEEN IN TRAINING!")
                        categorical_issues.append((feat, 'unseen_category'))
                    else:
                        # Show how common this category is in training
                        train_counts = (train_df[feat] == value).sum()
                        train_pct = (train_counts / len(train_df)) * 100
                        
                        # Get average sales for this category
                        if 'weekly_sales' in train_df.columns:
                            category_sales = train_df[train_df[feat] == value]['weekly_sales'].mean()
                            print(f"   ✓ {feat:35s}: {value} (seen {train_counts:,} times, {train_pct:.2f}%, "
                                  f"avg sales: {category_sales:.2f})")
                        else:
                            print(f"   ✓ {feat:35s}: {value} (seen {train_counts:,} times, {train_pct:.2f}%)")
                else:
                    print(f"   {feat:35s}: {value} | Feature not in training data")
        
        # ========== 3. MISSING/DEFAULT VALUES CHECK ==========
        print(f"\n{'─'*80}")
        print("3️⃣  MISSING/DEFAULT VALUES CHECK")
        print(f"{'─'*80}")
        print("   Checking for features with missing, zero, or default values...")
        
        # Get all numeric features that might have defaults
        numeric_features = test_df.select_dtypes(include=[np.number]).columns.tolist()
        missing_or_default = []
        
        for feat in numeric_features:
            if feat in product_features.index and feat not in ['ID']:
                value = product_features[feat]
                
                # Check for problematic values
                if pd.isna(value):
                    missing_or_default.append((feat, 'NaN'))
                elif value == 0 and feat not in ['is_week_23', 'is_black_friday', 'has_plus_sizes']:
                    # Zero might be problematic for most features (except binary flags)
                    missing_or_default.append((feat, 0))
                elif value == -1 and 'cluster' in feat.lower():
                    # -1 is often used as sentinel for invalid clusters
                    missing_or_default.append((feat, -1))
        
        if missing_or_default:
            print(f"   ⚠️  Found {len(missing_or_default)} features with missing/default values:")
            for feat, val in missing_or_default[:15]:  # Show first 15
                print(f"      - {feat}: {val}")
            if len(missing_or_default) > 15:
                print(f"      ... and {len(missing_or_default) - 15} more")
        else:
            print(f"   ✓ No obvious missing/default values detected")
        
        # ========== 4. SIMILAR PRODUCT COMPARISON ==========
        print(f"\n{'─'*80}")
        print("4️⃣  SIMILAR PRODUCT COMPARISON")
        print(f"{'─'*80}")
        print("   Finding products in training with similar characteristics...")
        
        if 'price' in product_features.index and 'num_stores' in product_features.index:
            price = product_features['price']
            num_stores = product_features['num_stores']
            
            if pd.notna(price) and pd.notna(num_stores):
                # Find similar products (within 20% of price and num_stores)
                similar = train_df[
                    (train_df['price'] >= price * 0.8) & (train_df['price'] <= price * 1.2) &
                    (train_df['num_stores'] >= num_stores * 0.8) & (train_df['num_stores'] <= num_stores * 1.2)
                ]
                
                if len(similar) > 0:
                    similar_sales = similar.groupby('ID')['weekly_sales'].sum()
                    print(f"   ✓ Found {len(similar)} similar products in training (within 20% of price & num_stores)")
                    print(f"   Their total sales distribution:")
                    print(f"      Min:    {similar_sales.min():,.2f}")
                    print(f"      25th:   {similar_sales.quantile(0.25):,.2f}")
                    print(f"      Median: {similar_sales.median():,.2f}")
                    print(f"      75th:   {similar_sales.quantile(0.75):,.2f}")
                    print(f"      Max:    {similar_sales.max():,.2f}")
                    print(f"      Mean:   {similar_sales.mean():,.2f}")
                    print(f"\n   Your prediction: {production_value}")
                    
                    if production_value < similar_sales.quantile(0.1):
                        print(f"   ⚠️  Your prediction is below the 10th percentile of similar products!")
                        print(f"   ⚠️  This suggests the prediction is unreasonably low")
                else:
                    print(f"   ⚠️  No similar products found in training!")
                    print(f"   ⚠️  This product might have a unique combination of features")
            else:
                print(f"   ⚠️  Cannot compare: price or num_stores is missing")
        else:
            print(f"   ⚠️  Cannot compare: price or num_stores not available")
        
        # Also compare by cluster if available
        if 'emb_cluster' in product_features.index:
            cluster = product_features['emb_cluster']
            if pd.notna(cluster) and cluster != -1:
                cluster_products = train_df[train_df['emb_cluster'] == cluster]
                if len(cluster_products) > 0:
                    cluster_sales = cluster_products.groupby('ID')['weekly_sales'].sum()
                    print(f"\n   Products in same emb_cluster ({cluster}):")
                    print(f"      Count:  {len(cluster_sales)} products")
                    print(f"      Median sales: {cluster_sales.median():,.2f}")
                    print(f"      Mean sales:   {cluster_sales.mean():,.2f}")
                    print(f"      Your prediction: {production_value}")
        
        # ========== 5. ROOT CAUSE ANALYSIS ==========
        print(f"\n{'─'*80}")
        print("5️⃣  ROOT CAUSE ANALYSIS")
        print(f"{'─'*80}")
        
        root_causes = []
        
        # Check for unseen categories
        if categorical_issues:
            unseen = [f for f, issue in categorical_issues if issue == 'unseen_category']
            if unseen:
                root_causes.append(f"Unseen categorical values: {', '.join(unseen)}")
        
        # Check for out-of-range features
        if feature_issues:
            out_of_range = [f for f, issue in feature_issues if issue == 'out_of_range']
            if out_of_range:
                root_causes.append(f"Out-of-range features: {', '.join(out_of_range)}")
        
        # Check for missing features
        missing = [f for f, issue in feature_issues if issue == 'missing']
        if missing:
            root_causes.append(f"Missing features: {', '.join(missing)}")
        
        # Check life cycle length
        if 'life_cycle_length' in product_features.index:
            life_cycle = product_features['life_cycle_length']
            if pd.notna(life_cycle) and life_cycle < 5:
                root_causes.append(f"Very short life cycle ({life_cycle} weeks) - small weekly predictions sum to < 0.5")
        
        # Check if product has very low velocity_1_3
        if 'velocity_1_3' in product_features.index:
            velocity = product_features['velocity_1_3']
            if pd.notna(velocity):
                train_velocity_q10 = train_df['velocity_1_3'].quantile(0.1)
                if velocity < train_velocity_q10:
                    root_causes.append(f"Very low velocity_1_3 ({velocity:.2f}) - cluster historically sells slowly")
        
        # Check if product has negative trend_score
        if 'trend_score' in product_features.index:
            trend = product_features['trend_score']
            if pd.notna(trend) and trend < -0.5:
                root_causes.append(f"Negative trend_score ({trend:.2f}) - similar to declining products")
        
        if root_causes:
            print("   Identified potential root causes:")
            for i, cause in enumerate(root_causes, 1):
                print(f"   {i}. {cause}")
        else:
            print("   ⚠️  No obvious root causes identified from feature analysis")
            print("   This might indicate:")
            print("      - Model is predicting negative weekly values that sum to < 0")
            print("      - Very small positive predictions that round to 0")
            print("      - Complex feature interactions causing low predictions")
        
        # ========== 6. RECOMMENDATIONS ==========
        print(f"\n{'─'*80}")
        print("6️⃣  RECOMMENDATIONS")
        print(f"{'─'*80}")
        
        recommendations = []
        
        if categorical_issues:
            recommendations.append("Fix categorical feature mismatches - ensure test categories exist in training")
        
        if feature_issues:
            recommendations.append("Review feature engineering - check if out-of-range values need special handling")
        
        if missing_or_default:
            recommendations.append("Review missing value imputation - default values might not be appropriate")
        
        if production_value == 0:
            recommendations.append("Apply minimum production threshold based on similar products or cluster averages")
            recommendations.append("Consider boosting very low predictions with adaptive penalization")
        
        if production_value == 1:
            recommendations.append("Product has minimal prediction - consider applying feature-based minimum")
            recommendations.append("Check if this is due to rounding (sum < 0.5) or actual model prediction")
        
        # General recommendations
        recommendations.append("Consider using feature-based minimums (cluster/family 10th percentile)")
        recommendations.append("Apply adaptive penalization: higher boost for very low predictions")
        recommendations.append("Review model training - ensure it handles edge cases properly")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print(f"\n{'='*80}\n")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Analyzed {len(problem_ids)} problematic products")
    print(f"Total problematic products: {len(problematic_ids)}")
    print(f"\nNext steps:")
    print(f"1. Review the root causes identified above")
    print(f"2. Implement fixes in train_model.py (minimum thresholds, adaptive penalization)")
    print(f"3. Re-run predictions and verify improvements")
    print(f"{'='*80}\n")


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Diagnose why products get 0 or 1 predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific products
  python diagnostic.py --ids 224 1691 3210
  
  # Analyze all problematic products
  python diagnostic.py --all-problematic
  
  # Analyze top 10 problematic products
  python diagnostic.py --top-n 10
        """
    )
    
    parser.add_argument('--ids', nargs='+', type=int, 
                       help='Specific product IDs to analyze')
    parser.add_argument('--all-problematic', action='store_true',
                       help='Analyze all products with Production = 0 or 1')
    parser.add_argument('--top-n', type=int,
                       help='Analyze top N problematic products')
    parser.add_argument('--predictions', default='outputs/predictions.csv',
                       help='Path to predictions.csv file')
    parser.add_argument('--test', default='data/processed/test_processed.csv',
                       help='Path to test_processed.csv file')
    parser.add_argument('--train', default='data/processed/train_processed.csv',
                       help='Path to train_processed.csv file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.ids is None and not args.all_problematic and args.top_n is None:
        # Default: analyze first 5 problematic products
        print("No specific products specified. Analyzing first 5 problematic products...")
        print("Use --ids, --all-problematic, or --top-n to specify which products to analyze\n")
    
    diagnose_zero_predictions(
        predictions_path=args.predictions,
        test_processed_path=args.test,
        train_processed_path=args.train,
        problem_ids=args.ids,
        analyze_all_problematic=args.all_problematic,
        top_n=args.top_n
    )


if __name__ == "__main__":
    main()

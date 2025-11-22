"""
Root Cause Analysis for Low Predictions (0 or 1)
This script identifies exactly why products get predicted to 0 or 1

Usage:
    python root_cause_analysis.py --id 224
    python root_cause_analysis.py --ids 224 1691 3210
    python root_cause_analysis.py --all-problematic
    python root_cause_analysis.py --top-n 10
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

def analyze_weekly_pattern(weekly_pred, product_id):
    """Analyze the weekly prediction pattern for a product"""
    product_weeks = weekly_pred[weekly_pred['ID'] == product_id].sort_values('weeks_since_launch')
    
    if len(product_weeks) == 0:
        return None
    
    total = product_weeks['predicted_weekly_sales'].sum()
    total_raw = product_weeks['predicted_weekly_sales_raw'].sum()
    
    analysis = {
        'product_id': product_id,
        'num_weeks': len(product_weeks),
        'total_production': total,
        'total_raw': total_raw,
        'weekly_breakdown': product_weeks[['weeks_since_launch', 'predicted_weekly_sales', 'predicted_weekly_sales_raw']].to_dict('records'),
        'avg_per_week': total / len(product_weeks) if len(product_weeks) > 0 else 0,
        'max_week': product_weeks['predicted_weekly_sales'].max(),
        'min_week': product_weeks['predicted_weekly_sales'].min(),
        'weeks_positive': (product_weeks['predicted_weekly_sales'] > 0).sum(),
        'weeks_negative': (product_weeks['predicted_weekly_sales'] < 0).sum(),
        'weeks_zero': (product_weeks['predicted_weekly_sales'] == 0).sum(),
        'weeks_very_small': (product_weeks['predicted_weekly_sales'] < 0.1).sum(),
    }
    
    return analysis

def root_cause_analysis(product_ids=None, analyze_all_problematic=False, top_n=None,
                        weekly_pred_path='outputs/weekly_predictions_debug.csv',
                        predictions_path='outputs/predictions.csv',
                        test_path='data/processed/test_processed.csv',
                        train_path='data/processed/train_processed.csv'):
    """
    Comprehensive root cause analysis for products with 0 or 1 predictions
    """
    print("="*100)
    print("ROOT CAUSE ANALYSIS FOR LOW PREDICTIONS (0 or 1)")
    print("="*100)
    
    # Check if weekly predictions file exists
    if not Path(weekly_pred_path).exists():
        print(f"\n❌ ERROR: {weekly_pred_path} not found!")
        print("   You need to run train_model.py first with save_weekly_debug=True")
        print("   The weekly predictions file is created during model prediction.")
        return
    
    # Load data
    print("\nLoading data files...")
    weekly_pred = pd.read_csv(weekly_pred_path)
    predictions_df = pd.read_csv(predictions_path)
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    
    print(f"✓ Loaded weekly predictions: {len(weekly_pred):,} rows")
    print(f"✓ Loaded final predictions: {len(predictions_df):,} products")
    print(f"✓ Loaded test data: {len(test_df):,} products")
    print(f"✓ Loaded training data: {len(train_df):,} rows")
    
    # Identify problematic products
    if analyze_all_problematic:
        problematic_ids = predictions_df[
            (predictions_df['Production'] == 0) | (predictions_df['Production'] == 1)
        ]['ID'].tolist()
        print(f"\n📊 Found {len(problematic_ids)} products with Production = 0 or 1")
    elif top_n:
        problematic_df = predictions_df[
            (predictions_df['Production'] == 0) | (predictions_df['Production'] == 1)
        ].head(top_n)
        problematic_ids = problematic_df['ID'].tolist()
        print(f"\n📊 Analyzing top {top_n} products with Production = 0 or 1")
    elif product_ids:
        problematic_ids = product_ids
        print(f"\n📊 Analyzing {len(problematic_ids)} specified products")
    else:
        print("\n❌ Please specify --id, --ids, --all-problematic, or --top-n")
        return
    
    # Analyze each problematic product
    for idx, product_id in enumerate(problematic_ids, 1):
        print(f"\n{'='*100}")
        print(f"PRODUCT {idx}/{len(problematic_ids)}: ID {product_id}")
        print(f"{'='*100}")
        
        # Get final prediction
        production = predictions_df[predictions_df['ID'] == product_id]['Production'].values
        production = production[0] if len(production) > 0 else None
        
        print(f"\n📊 FINAL PREDICTION: {production}")
        
        # Weekly breakdown analysis
        weekly_analysis = analyze_weekly_pattern(weekly_pred, product_id)
        
        if not weekly_analysis:
            print(f"   ⚠️  Product {product_id} not found in weekly predictions")
            continue
        
        print(f"\n📅 WEEKLY PREDICTION BREAKDOWN:")
        print(f"   Number of weeks: {weekly_analysis['num_weeks']}")
        print(f"   Total (after penalization): {weekly_analysis['total_production']:.4f}")
        print(f"   Total (raw, before penalization): {weekly_analysis['total_raw']:.4f}")
        print(f"   Average per week: {weekly_analysis['avg_per_week']:.4f}")
        print(f"   Max weekly prediction: {weekly_analysis['max_week']:.4f}")
        print(f"   Min weekly prediction: {weekly_analysis['min_week']:.4f}")
        print(f"   Weeks with positive predictions: {weekly_analysis['weeks_positive']}")
        print(f"   Weeks with negative predictions: {weekly_analysis['weeks_negative']}")
        print(f"   Weeks with zero predictions: {weekly_analysis['weeks_zero']}")
        print(f"   Weeks with very small predictions (<0.1): {weekly_analysis['weeks_very_small']}")
        
        # Show weekly breakdown table
        print(f"\n   Weekly breakdown (first 15 weeks):")
        print(f"   {'Week':<6} {'Raw Pred':<12} {'After Penal':<12} {'Cumulative':<12}")
        print(f"   {'-'*45}")
        cumulative = 0
        for week_data in weekly_analysis['weekly_breakdown'][:15]:
            cumulative += week_data['predicted_weekly_sales']
            print(f"   {int(week_data['weeks_since_launch']):<6} "
                  f"{week_data['predicted_weekly_sales_raw']:<12.4f} "
                  f"{week_data['predicted_weekly_sales']:<12.4f} "
                  f"{cumulative:<12.4f}")
        
        if weekly_analysis['num_weeks'] > 15:
            print(f"   ... ({weekly_analysis['num_weeks'] - 15} more weeks)")
        
        # Root cause identification
        print(f"\n🔍 ROOT CAUSE IDENTIFICATION:")
        root_causes = []
        
        if weekly_analysis['total_raw'] < 0:
            print(f"   ❌ CRITICAL: Total raw prediction is NEGATIVE ({weekly_analysis['total_raw']:.4f})")
            print(f"      → Model is predicting negative sales, which gets clipped to 0")
            print(f"      → This suggests features are pushing prediction below zero")
            root_causes.append("Negative total prediction")
            
            if weekly_analysis['weeks_negative'] > 0:
                print(f"      → {weekly_analysis['weeks_negative']} weeks have negative predictions")
                negative_weeks = [w for w in weekly_analysis['weekly_breakdown'] 
                                 if w['predicted_weekly_sales'] < 0]
                print(f"      → Negative weeks: {[int(w['weeks_since_launch']) for w in negative_weeks[:5]]}")
        
        elif weekly_analysis['total_raw'] < 0.5:
            print(f"   ❌ CRITICAL: Total raw prediction is very small ({weekly_analysis['total_raw']:.4f})")
            print(f"      → After rounding, this becomes 0")
            print(f"      → All weekly predictions are too small")
            root_causes.append("Very small total prediction (< 0.5)")
        
        elif weekly_analysis['total_production'] < 1.5:
            print(f"   ⚠️  WARNING: Total prediction is very low ({weekly_analysis['total_production']:.4f})")
            print(f"      → Rounds to {int(round(weekly_analysis['total_production']))}")
            root_causes.append("Low total prediction")
        
        if weekly_analysis['weeks_negative'] > 0:
            print(f"   ⚠️  WARNING: {weekly_analysis['weeks_negative']} weeks have negative predictions")
            print(f"      → These negative weeks are reducing the total")
            root_causes.append(f"{weekly_analysis['weeks_negative']} negative weeks")
        
        if weekly_analysis['avg_per_week'] < 0.1:
            print(f"   ⚠️  WARNING: Average per week is very low ({weekly_analysis['avg_per_week']:.4f})")
            print(f"      → Even with {weekly_analysis['num_weeks']} weeks, total is too small")
            root_causes.append("Very low average per week")
        
        if weekly_analysis['num_weeks'] < 5:
            print(f"   ⚠️  WARNING: Very short life cycle ({weekly_analysis['num_weeks']} weeks)")
            print(f"      → Short life cycles make it harder to accumulate enough sales")
            root_causes.append("Short life cycle")
        
        if weekly_analysis['weeks_very_small'] == weekly_analysis['num_weeks']:
            print(f"   ⚠️  WARNING: ALL weeks have very small predictions (< 0.1)")
            print(f"      → Model consistently predicts very low sales for this product")
            root_causes.append("All weeks very small")
        
        # Feature analysis
        product = test_df[test_df['ID'] == product_id]
        if len(product) > 0:
            product = product.iloc[0]
            
            print(f"\n📋 KEY FEATURE ANALYSIS:")
            key_features = [
                ('price', 'Price'),
                ('num_stores', 'Number of Stores'),
                ('num_sizes', 'Number of Sizes'),
                ('life_cycle_length', 'Life Cycle Length'),
                ('velocity_1_3', 'Velocity (first 3 weeks)'),
                ('trend_score', 'Trend Score'),
                ('sim_to_top', 'Similarity to Top Performers'),
                ('sim_to_bottom', 'Similarity to Bottom Performers'),
                ('emb_cluster', 'Embedding Cluster'),
                ('emb_dist', 'Distance to Cluster Center'),
                ('color_cluster', 'Color Cluster'),
                ('family', 'Family'),
            ]
            
            problematic_features = []
            
            for feat, label in key_features:
                if feat in product.index:
                    value = product[feat]
                    
                    # Compare with training
                    if feat in train_df.columns:
                        train_mean = train_df[feat].mean()
                        train_median = train_df[feat].median()
                        train_q10 = train_df[feat].quantile(0.1)
                        train_q90 = train_df[feat].quantile(0.9)
                        
                        if pd.notna(value):
                            if isinstance(value, (int, float)):
                                if value < train_q10:
                                    flag = "⬇️  VERY LOW"
                                    problematic_features.append((feat, value, train_median, "very_low"))
                                elif value < train_median * 0.7:
                                    flag = "⬇️  LOW"
                                    problematic_features.append((feat, value, train_median, "low"))
                                elif value > train_q90:
                                    flag = "⬆️  VERY HIGH"
                                    problematic_features.append((feat, value, train_median, "very_high"))
                                elif value > train_median * 1.3:
                                    flag = "⬆️  HIGH"
                                else:
                                    flag = "✓ NORMAL"
                                
                                print(f"   {label:30s}: {value:>10.2f} {flag:15s} "
                                      f"(train median: {train_median:.2f})")
                            else:
                                # Categorical
                                train_categories = set(train_df[feat].dropna().unique())
                                if value not in train_categories:
                                    flag = "⚠️  UNSEEN"
                                    problematic_features.append((feat, value, "N/A", "unseen_category"))
                                else:
                                    flag = "✓"
                                
                                print(f"   {label:30s}: {value:>10} {flag:15s}")
            
            # Summary of problematic features
            if problematic_features:
                print(f"\n⚠️  PROBLEMATIC FEATURES DETECTED:")
                for feat, value, train_ref, issue_type in problematic_features:
                    if issue_type == "very_low":
                        print(f"   • {feat}: {value:.2f} (very low, median: {train_ref:.2f})")
                    elif issue_type == "low":
                        print(f"   • {feat}: {value:.2f} (low, median: {train_ref:.2f})")
                    elif issue_type == "very_high":
                        print(f"   • {feat}: {value:.2f} (very high, median: {train_ref:.2f})")
                    elif issue_type == "unseen_category":
                        print(f"   • {feat}: {value} (unseen category in training)")
        
        # Similar products comparison
        if 'price' in product.index and 'num_stores' in product.index and 'emb_cluster' in product.index:
            price = product['price']
            num_stores = product['num_stores']
            emb_cluster = product['emb_cluster']
            
            print(f"\n🔍 SIMILAR PRODUCTS COMPARISON:")
            
            # Filter similar products
            similar = train_df.copy()
            
            # Same cluster if available
            if pd.notna(emb_cluster) and emb_cluster >= 0:
                similar = similar[similar['emb_cluster'] == emb_cluster]
                print(f"   Filtering by same emb_cluster ({int(emb_cluster)})")
            
            # Similar price (within 30%)
            if pd.notna(price):
                similar = similar[
                    (similar['price'] >= price * 0.7) & 
                    (similar['price'] <= price * 1.3)
                ]
                print(f"   Filtering by similar price (within 30% of {price:.2f})")
            
            # Similar num_stores (within 30%)
            if pd.notna(num_stores):
                similar = similar[
                    (similar['num_stores'] >= num_stores * 0.7) & 
                    (similar['num_stores'] <= num_stores * 1.3)
                ]
                print(f"   Filtering by similar num_stores (within 30% of {int(num_stores)})")
            
            if len(similar) > 0:
                # Group by product ID and get total sales
                similar_grouped = similar.groupby('ID').agg({
                    'weekly_sales': 'sum',
                    'price': 'first',
                    'num_stores': 'first',
                    'num_sizes': 'first',
                    'family': 'first'
                }).reset_index()
                
                similar_grouped = similar_grouped.sort_values('weekly_sales', ascending=False).head(10)
                
                print(f"\n   Found {len(similar)} similar product-week combinations in training")
                print(f"   Top 10 similar products by total sales:")
                print(f"   {'ID':<10} {'Total Sales':<15} {'Price':<10} {'Stores':<10} {'Sizes':<10} {'Family':<20}")
                print(f"   {'-'*90}")
                
                for _, row in similar_grouped.iterrows():
                    print(f"   {int(row['ID']):<10} {row['weekly_sales']:<15.0f} "
                          f"{row['price']:<10.2f} {int(row['num_stores']):<10} "
                          f"{int(row['num_sizes']):<10} {str(row['family']):<20}")
                
                similar_avg = similar_grouped['weekly_sales'].mean()
                similar_median = similar_grouped['weekly_sales'].median()
                
                print(f"\n   📊 Similar products statistics:")
                print(f"      Average total sales: {similar_avg:.0f}")
                print(f"      Median total sales:  {similar_median:.0f}")
                print(f"      Min total sales:     {similar_grouped['weekly_sales'].min():.0f}")
                print(f"      Max total sales:     {similar_grouped['weekly_sales'].max():.0f}")
                
                if production is not None:
                    print(f"\n   💡 YOUR PREDICTION ({production}) vs SIMILAR PRODUCTS:")
                    diff_pct = ((production - similar_avg) / similar_avg * 100) if similar_avg > 0 else 0
                    if diff_pct < -80:
                        print(f"      ❌ Your prediction is {abs(diff_pct):.1f}% LOWER than similar products average")
                        print(f"      ❌ This is a strong indicator of under-prediction")
                    elif diff_pct < -50:
                        print(f"      ⚠️  Your prediction is {abs(diff_pct):.1f}% LOWER than similar products average")
                    elif diff_pct > 50:
                        print(f"      ⚠️  Your prediction is {diff_pct:.1f}% HIGHER than similar products average")
                    else:
                        print(f"      ✓ Your prediction is within reasonable range ({diff_pct:+.1f}%)")
            else:
                print(f"   ⚠️  No similar products found in training data")
                print(f"   ⚠️  This product might have a unique combination of features")
        
        # Summary
        print(f"\n📝 SUMMARY:")
        print(f"   Final Prediction: {production}")
        print(f"   Root Causes: {', '.join(root_causes) if root_causes else 'None identified'}")
        if problematic_features:
            print(f"   Problematic Features: {len(problematic_features)}")
        
        print(f"\n{'─'*100}")
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*100}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Root cause analysis for low predictions (0 or 1)')
    parser.add_argument('--id', type=int, help='Single product ID to analyze')
    parser.add_argument('--ids', type=int, nargs='+', help='Multiple product IDs to analyze')
    parser.add_argument('--all-problematic', action='store_true', 
                       help='Analyze all products with Production = 0 or 1')
    parser.add_argument('--top-n', type=int, 
                       help='Analyze top N products with Production = 0 or 1')
    args = parser.parse_args()
    
    # Determine which products to analyze
    product_ids = None
    if args.id:
        product_ids = [args.id]
    elif args.ids:
        product_ids = args.ids
    
    root_cause_analysis(
        product_ids=product_ids,
        analyze_all_problematic=args.all_problematic,
        top_n=args.top_n
    )


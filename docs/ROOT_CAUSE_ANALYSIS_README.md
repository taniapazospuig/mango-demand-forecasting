# Root Cause Analysis for Low Predictions

This guide explains how to diagnose why products are being predicted to 0 or 1.

## Overview

The model predicts **weekly sales** for each product, then sums them to get the final **Production** value. When a product gets 0 or 1, it's because:

1. **Negative weekly predictions** - Model predicts negative sales (gets clipped to 0)
2. **Very small weekly predictions** - All weekly predictions are too small, sum to < 0.5 (rounds to 0)
3. **Short life cycle** - Product has very few weeks, so even normal weekly predictions sum to a small total
4. **Problematic features** - Product has features that push predictions down (e.g., very high price, very low num_stores)

## Setup

1. **Run model training/prediction** - The weekly predictions are automatically saved when you run `train_model.py`:
   ```bash
   python train_model.py
   ```
   This creates `weekly_predictions_debug.csv` with the weekly breakdown.

2. **Run root cause analysis** - Use the analysis script to investigate specific products:
   ```bash
   # Analyze a single product
   python root_cause_analysis.py --id 224
   
   # Analyze multiple products
   python root_cause_analysis.py --ids 224 1691 3210
   
   # Analyze all problematic products (0 or 1)
   python root_cause_analysis.py --all-problematic
   
   # Analyze top N problematic products
   python root_cause_analysis.py --top-n 10
   ```

## What the Analysis Shows

For each product, the analysis provides:

### 1. Weekly Prediction Breakdown
- Total prediction (raw and after penalization)
- Number of weeks
- Average per week
- Max/min weekly predictions
- Count of positive/negative/zero weeks
- **Weekly breakdown table** showing predictions for each week

### 2. Root Cause Identification
Automatically identifies:
- ❌ **Negative total prediction** - Model predicts negative sales
- ❌ **Very small total** - Sum < 0.5, rounds to 0
- ⚠️ **Negative weeks** - Some weeks have negative predictions
- ⚠️ **Very low average** - Average per week is too small
- ⚠️ **Short life cycle** - Too few weeks to accumulate sales
- ⚠️ **All weeks very small** - Model consistently predicts low sales

### 3. Feature Analysis
Compares product features to training data:
- Shows if features are **VERY LOW**, **LOW**, **NORMAL**, **HIGH**, or **VERY HIGH**
- Flags **unseen categories** (not in training data)
- Identifies **problematic features** that might cause low predictions

### 4. Similar Products Comparison
Finds products in training data with:
- Same embedding cluster
- Similar price (within 30%)
- Similar number of stores (within 30%)

Shows their actual sales and compares to your prediction.

## Example Output

```
====================================================================================================
PRODUCT 1/5: ID 224
====================================================================================================

📊 FINAL PREDICTION: 0

📅 WEEKLY PREDICTION BREAKDOWN:
   Number of weeks: 12
   Total (after penalization): 0.2341
   Total (raw, before penalization): 0.2148
   Average per week: 0.0195
   Max weekly prediction: 0.0456
   Min weekly prediction: -0.0123
   Weeks with positive predictions: 10
   Weeks with negative predictions: 2
   Weeks with zero predictions: 0

   Weekly breakdown (first 15 weeks):
   Week   Raw Pred     After Penal  Cumulative  
   ---------------------------------------------
   1      0.0234       0.0255       0.0255
   2      0.0312       0.0340       0.0595
   ...

🔍 ROOT CAUSE IDENTIFICATION:
   ❌ CRITICAL: Total raw prediction is very small (0.2148)
      → After rounding, this becomes 0
      → All weekly predictions are too small
   ⚠️  WARNING: 2 weeks have negative predictions
      → These negative weeks are reducing the total

📋 KEY FEATURE ANALYSIS:
   Price                        :     89.99 ⬇️  VERY LOW    (train median: 45.50)
   Number of Stores             :         5 ⬇️  VERY LOW    (train median: 120.00)
   ...

⚠️  PROBLEMATIC FEATURES DETECTED:
   • price: 89.99 (very high, median: 45.50)
   • num_stores: 5 (very low, median: 120.00)

🔍 SIMILAR PRODUCTS COMPARISON:
   Found 45 similar product-week combinations in training
   Top 10 similar products by total sales:
   ID         Total Sales     Price      Stores     Sizes       Family              
   ------------------------------------------------------------------------------------------
   1234       1250           42.50      8          6           T-SHIRT
   ...

   💡 YOUR PREDICTION (0) vs SIMILAR PRODUCTS:
      ❌ Your prediction is 95.2% LOWER than similar products average
      ❌ This is a strong indicator of under-prediction
```

## Common Root Causes

1. **Very High Price** - Products with prices in top 10% of training data often get low predictions
2. **Very Low num_stores** - Products with < 10 stores get very low predictions
3. **Negative trend_score** - Products similar to bottom performers get penalized
4. **Unseen Categories** - New categories not in training data confuse the model
5. **Short Life Cycle** - Products with < 5 weeks can't accumulate enough sales
6. **Negative Weekly Predictions** - Some weeks predict negative sales, reducing total

## Next Steps

After identifying root causes, you can:

1. **Adjust features** - If features are problematic, check data quality
2. **Modify model** - Add minimum thresholds or feature-based adjustments
3. **Improve training** - Add more examples of edge cases to training data
4. **Post-processing** - Apply business rules to handle edge cases

## Files

- `train_model.py` - Modified to save `weekly_predictions_debug.csv`
- `root_cause_analysis.py` - Main analysis script
- `weekly_predictions_debug.csv` - Weekly predictions breakdown (created by train_model.py)
- `predictions.csv` - Final predictions
- `test_processed.csv` - Test data with features
- `train_processed.csv` - Training data with features


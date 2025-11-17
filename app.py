"""Streamlit app for Mango Demand Forecasting."""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from src.data.preprocessing import load_and_prepare_train_data, process_test_data
from src.models.predictor import predict_production
from src.utils.config import PENALIZATION_FACTOR

# Page configuration
st.set_page_config(
    page_title="Mango Demand Forecasting",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #E55A2B;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model_data():
    """Load model and data with caching."""
    try:
        # Check if processed data exists
        if not os.path.exists('data/processed/train_processed.csv'):
            return None, None, None, "Processed training data not found. Please run data preprocessing first."
        
        # Load training data to get feature info
        X, y, categorical_features, seasons, bin_info = load_and_prepare_train_data(
            'data/processed/train_processed.csv', return_seasons=True
        )
        
        # Try to load saved models
        models = []
        models_dir = 'models/checkpoints'
        if os.path.exists(models_dir):
            for i in range(5):  # Try to load up to 5 models
                model_path = os.path.join(models_dir, f'model_{i}.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        models.append(pickle.load(f))
        
        if not models:
            return X, y, categorical_features, bin_info, "No saved models found. Please train models first using train.py"
        
        return X, y, categorical_features, bin_info, None
    except Exception as e:
        return None, None, None, f"Error loading data: {str(e)}"


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🥭 Mango Demand Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("### Model Settings")
        
        penalization_factor = st.slider(
            "Penalization Factor",
            min_value=1.0,
            max_value=1.5,
            value=PENALIZATION_FACTOR,
            step=0.01,
            help="Factor to multiply predictions by to avoid being short on demand"
        )
        
        max_weeks = st.slider(
            "Max Weeks per Product",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Maximum number of weeks to predict for each product"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This app predicts weekly demand for fashion products using a LightGBM ensemble model.
        
        **Features:**
        - Ensemble of 5 models
        - Time-based cross-validation
        - Advanced feature engineering
        - Production forecasting
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Predictions", "📈 Analytics", "ℹ️ Info"])
    
    with tab1:
        st.header("Model Overview")
        
        # Load data
        X, y, categorical_features, bin_info, error = load_model_data()
        
        if error:
            st.error(error)
            st.info("""
            **To get started:**
            1. Ensure `data/processed/train_processed.csv` and `data/processed/test_processed.csv` exist
            2. Run `python train.py` to train models
            3. Models will be saved to `models/checkpoints/`
            4. Refresh this page
            """)
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Training Samples", f"{len(X):,}")
            
            with col2:
                st.metric("Features", len(X.columns))
            
            with col3:
                st.metric("Categorical Features", len(categorical_features))
            
            with col4:
                st.metric("Target Mean", f"{y.mean():.2f}")
            
            st.markdown("---")
            
            # Feature importance if available
            if os.path.exists('outputs/feature_importance.csv'):
                st.subheader("Top 20 Most Important Features")
                feature_imp = pd.read_csv('outputs/feature_importance.csv')
                top_features = feature_imp.head(20)
                
                st.bar_chart(top_features.set_index('feature')['importance'])
                
                with st.expander("View All Features"):
                    st.dataframe(feature_imp, use_container_width=True)
    
    with tab2:
        st.header("Generate Predictions")
        
        X, y, categorical_features, bin_info, error = load_model_data()
        
        if error:
            st.error(error)
        else:
            if st.button("🚀 Generate Predictions", use_container_width=True):
                with st.spinner("Loading test data and generating predictions..."):
                    try:
                        # Load models
                        models = []
                        models_dir = 'models/checkpoints'
                        for i in range(5):
                            model_path = os.path.join(models_dir, f'model_{i}.pkl')
                            if os.path.exists(model_path):
                                with open(model_path, 'rb') as f:
                                    models.append(pickle.load(f))
                        
                        if not models:
                            st.error("No models found. Please train models first.")
                            st.stop()
                        
                        # Process test data
                        X_test, df_meta = process_test_data(
                            'data/processed/test_processed.csv',
                            max_weeks=max_weeks,
                            train_categorical_cols=categorical_features,
                            bin_info=bin_info
                        )
                        
                        # Ensure feature alignment
                        for col in X.columns:
                            if col not in X_test.columns:
                                if col in categorical_features:
                                    if col in ['num_stores_bin', 'num_sizes_bin']:
                                        most_common = X[col].value_counts().index[0]
                                        X_test[col] = pd.Categorical(
                                            [most_common] * len(X_test),
                                            categories=X[col].cat.categories
                                        )
                                    else:
                                        X_test[col] = pd.Categorical(['MISSING'] * len(X_test))
                                else:
                                    X_test[col] = 0
                        
                        X_test = X_test[X.columns]
                        
                        # Make predictions
                        production_df = predict_production(
                            models, X_test, df_meta, penalization_factor, save_weekly_debug=False
                        )
                        
                        # Save predictions
                        os.makedirs('outputs', exist_ok=True)
                        production_df.to_csv('outputs/predictions.csv', index=False)
                        
                        st.success("✅ Predictions generated successfully!")
                        
                        # Display summary
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Products", len(production_df))
                        
                        with col2:
                            st.metric("Total Production", f"{production_df['Production'].sum():,}")
                        
                        with col3:
                            st.metric("Avg Production", f"{production_df['Production'].mean():.2f}")
                        
                        with col4:
                            st.metric("Max Production", f"{production_df['Production'].max():,}")
                        
                        st.markdown("---")
                        
                        # Display predictions table
                        st.subheader("Predictions Preview")
                        st.dataframe(production_df.head(100), use_container_width=True)
                        
                        # Download button
                        csv = production_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
                        st.exception(e)
            
            # Show existing predictions if available
            if os.path.exists('outputs/predictions.csv'):
                st.markdown("---")
                st.subheader("Existing Predictions")
                existing_preds = pd.read_csv('outputs/predictions.csv')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Products", len(existing_preds))
                with col2:
                    st.metric("Total Production", f"{existing_preds['Production'].sum():,}")
                with col3:
                    st.metric("Avg Production", f"{existing_preds['Production'].mean():.2f}")
                
                st.dataframe(existing_preds, use_container_width=True)
    
    with tab3:
        st.header("Analytics & Insights")
        
        if os.path.exists('outputs/predictions.csv'):
            preds = pd.read_csv('outputs/predictions.csv')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Production Distribution")
                st.histogram_chart(preds['Production'])
            
            with col2:
                st.subheader("Production Statistics")
                stats = preds['Production'].describe()
                st.dataframe(stats)
            
            st.markdown("---")
            
            # Top and bottom products
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Products by Production")
                top_products = preds.nlargest(10, 'Production')
                st.dataframe(top_products, use_container_width=True)
            
            with col2:
                st.subheader("Bottom 10 Products by Production")
                bottom_products = preds.nsmallest(10, 'Production')
                st.dataframe(bottom_products, use_container_width=True)
        else:
            st.info("No predictions available. Generate predictions first.")
    
    with tab4:
        st.header("Project Information")
        
        st.markdown("""
        ### About This Project
        
        This is a **Mango Demand Forecasting** system that predicts weekly sales for fashion products
        using machine learning.
        
        ### Model Architecture
        
        - **Algorithm**: LightGBM (Gradient Boosting)
        - **Ensemble**: 5 models with different random seeds
        - **Validation**: Time-based cross-validation (seasons 86-88 vs 89)
        
        ### Key Features
        
        1. **Feature Engineering**
           - Binned numeric features (num_stores, num_sizes)
           - Cluster-based features (embedding clusters, color clusters)
           - Trend features (similarity to top/bottom performers)
           - Seasonality features (week 23, Black Friday)
           - Family and cluster-level aggregations
        
        2. **Prediction Pipeline**
           - Weekly sales prediction per product
           - Aggregation to total production per product
           - Penalization factor to avoid stockouts
        
        3. **Model Training**
           - Optional hyperparameter tuning with Optuna
           - Ensemble averaging for robustness
           - Feature importance analysis
        
        ### Usage
        
        1. **Preprocess Data**: Run `data_exploration.ipynb` to create processed datasets
        2. **Train Models**: Run `python train.py` to train the ensemble
        3. **Generate Predictions**: Use this app or run `train.py` directly
        4. **Analyze Results**: Check predictions.csv and feature_importance.csv
        
        ### Configuration
        
        Edit `src/utils/config.py` to adjust:
        - Penalization factor
        - Number of ensemble models
        - Hyperparameter tuning settings
        - Feature exclusions
        """)
        
        st.markdown("---")
        st.markdown("### Technical Details")
        
        if os.path.exists('outputs/feature_importance.csv'):
            st.subheader("Feature Importance")
            feature_imp = pd.read_csv('outputs/feature_importance.csv')
            st.dataframe(feature_imp.head(30), use_container_width=True)


if __name__ == "__main__":
    main()


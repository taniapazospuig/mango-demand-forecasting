# MANGO Demand Forecasting

Mango demand forecasting solution for FME Datathon 2025. This project predicts demand (maximum potential sales with infinite stock) for fashion products in a new season, providing production quantities to send to suppliers. The solution uses historical data from four seasons (two full years plus validation), containing product image embeddings, product family and attributes, phase information, number of stores and sizes, and past sales and production data. An ensemble of LightGBM models is trained to forecast demand, with the evaluation metric penalizing lost sales more heavily than excess stock, reflecting real-world retail constraints.

## Authors

Jan Aguiló, Júlia López, Tània Pazos, and Aniol Petit

## Project Structure

```
mango-demand-forecasting/
├── src/                          # Source code modules
│   └── utils/                    # Configuration
├── data/                         # Data files
├── data_exploration.ipynb        # Data preprocessing notebook
├── models/                       # Saved models
├── outputs/                      # Predictions and feature importance
├── train.py                      # Main training script
├── app.py                        # Streamlit application
└── requirements.txt              # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mango-demand-forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preprocessing

Run the Jupyter notebook to preprocess the raw data:

```bash
jupyter notebook data_exploration.ipynb
```

This notebook cleans and transforms the raw data, engineers features (clustering, trend features, etc.), and creates `data/processed/train_processed.csv` and `data/processed/test_processed.csv`.

### Step 2: Train Models

Train the ensemble model:

```bash
python train.py
```

This will:
- Load processed training data
- Perform time-based cross-validation
- Train an ensemble of 5 LightGBM models
- Save models to `models/checkpoints/`
- Generate predictions on test data
- Save results to `outputs/predictions.csv` and `outputs/feature_importance.csv`

### Step 3: Use Streamlit App (Optional)

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

## Model Architecture

- **Algorithm**: LightGBM (Gradient Boosting Decision Trees)
- **Ensemble**: 5 models averaged together
- **Validation**: Time-based cross-validation (seasons 86-88 for train, 89 for validation)
- **Objective**: Regression (MSE loss)

## Key Features

- Image embedding clustering to identify visually similar products
- Color clustering and trend analysis
- Lag features and moving averages for temporal patterns
- Seasonality indicators (Black Friday, peak weeks)
- Cluster and family-level aggregations
- Weekly demand forecasting aggregated to total production requirements

## Configuration

Edit `src/utils/config.py` to customize model parameters, penalization factors, and hyperparameter tuning settings.

## Output Files

After running `train.py`, you'll get:

- `outputs/predictions.csv`: Final production predictions (ID, Production)
- `outputs/feature_importance.csv`: Feature importance rankings
- `models/checkpoints/model_*.pkl`: Saved model files

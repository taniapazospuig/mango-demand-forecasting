# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Preprocess Data

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

This creates:
- `data/processed/train_processed.csv`
- `data/processed/test_processed.csv`

### Step 3: Train Models

```bash
python train.py
```

This will:
- Train 5 ensemble models
- Save models to `models/checkpoints/`
- Generate `outputs/predictions.csv`

### Step 4: Launch Streamlit App (Optional)

```bash
streamlit run app.py
```

Open your browser to the URL shown (usually http://localhost:8501)

## 📋 Workflow Summary

```
Raw Data (data/train.csv, data/test.csv)
    ↓
[Data Preprocessing] (notebooks/data_exploration.ipynb)
    ↓
Processed Data (data/processed/train_processed.csv, data/processed/test_processed.csv)
    ↓
[Model Training] (python train.py)
    ↓
Trained Models (models/checkpoints/model_*.pkl)
    ↓
[Prediction] (train.py or app.py)
    ↓
Predictions (predictions.csv)
```

## ⚡ Quick Tips

- **First time?** Start with default settings in `src/utils/config.py`
- **Fast iteration?** Set `N_ENSEMBLE_MODELS = 1` in config
- **Better results?** Enable hyperparameter tuning (slower)
- **Interactive?** Use the Streamlit app for exploration

## 🔧 Configuration

Edit `src/utils/config.py` to customize:
- `PENALIZATION_FACTOR`: Adjust prediction multiplier
- `N_ENSEMBLE_MODELS`: Number of models (1-5 recommended)
- `ENABLE_HYPERPARAM_TUNING`: Enable/disable tuning

## 📊 Output Files

After training, you'll have:
- `outputs/predictions.csv` - Final predictions
- `outputs/feature_importance.csv` - Feature rankings
- `models/checkpoints/model_*.pkl` - Saved models

## ❓ Troubleshooting

**"Processed data not found"**
→ Run `notebooks/data_exploration.ipynb` first (creates files in `data/processed/`)

**"No models found"**
→ Run `python train.py` to train models

**"Feature mismatch"**
→ Ensure test data was processed the same way as training data

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the Streamlit app for interactive analysis
- Check feature importance to understand model behavior


# Residential Appliance Energy Consumption Prediction

An end-to-end machine learning regression project using **19,735 observations** of household energy consumption recorded at 10-minute intervals. The project emphasizes time-aware evaluation, feature engineering, hyperparameter tuning, model interpretation, and honest reporting of generalization limitations.

## Problem

Predict `Appliances` energy consumption (Wh) using indoor temperature/humidity measurements, outdoor weather information, lighting, and engineered calendar features.

## Dataset

**UCI Machine Learning Repository — Appliances Energy Prediction**: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

Download `energydata_complete.csv` from the source and place it at `data/energydata_complete.csv`. The dataset is not included in this repository.

## Methodology

1. **EDA and quality checks:** inspected missing values, duplicates, distributions, outliers, correlations, and temporal patterns.
2. **Feature engineering:** extracted hour, weekday, month, and weekend indicators; encoded hour and weekday cyclically using sine/cosine.
3. **Temporal evaluation:** chronological 70% training / 15% validation / 15% test split; five-fold `TimeSeriesSplit` within training for tuning.
4. **Models:** mean baseline, Linear Regression, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, and Scikit-learn Gradient Boosting.
5. **Tuning:** GridSearchCV and RandomizedSearchCV, selecting models using held-out validation RMSE.
6. **Interpretability:** residual analysis, tree feature importance, and validation-set permutation importance.
7. **Final evaluation:** selected model evaluated once on the previously untouched test period.

## Results

| Metric | Validation | Final test |
|---|---:|---:|
| MAE (Wh) | 48.07 | 63.49 |
| RMSE (Wh) | 84.87 | 90.42 |
| R² | 0.153 | 0.010 |

The tuned `GradientBoostingRegressor` was selected because it had the lowest held-out validation RMSE among evaluated models. Its test R² of **0.010** indicates only marginal improvement in squared error compared with predicting the test-period mean. Residual analysis showed frequent underprediction of high-consumption events. **This model is not presented as deployment-ready.**

### Selected validation RMSE comparison

| Model | Validation RMSE (Wh) |
|---|---:|
| Mean baseline | 92.39 |
| Linear Regression | 86.12 |
| Tuned Ridge | 85.07 |
| Tuned Decision Tree | 89.04 |
| Tuned Random Forest | 86.20 |
| **Tuned Gradient Boosting** | **84.87** |

## Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook appliance_energy_prediction.ipynb
```

Download the CSV into `data/` before running. In Jupyter, select **Restart Kernel and Run All**. Hyperparameter searches may take time. Reported results come from the original analysis; **the cleaned, portable notebook still needs a full reproducibility run**.

## Key lessons

- Time-aware evaluation is critical for sequential observations.
- High training or validation performance does not guarantee performance on future periods.
- High-demand events are difficult to predict with the available inputs.
- Permutation importance describes predictive reliance, not causality.

## Potential future work

Evaluate lagged consumption features (without leakage), stronger temporal baselines, additional usage data, and fresh chronological holdouts. These are proposed extensions, not completed experiments.

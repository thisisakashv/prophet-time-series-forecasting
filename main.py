import pandas as pd
import numpy as np
from prophet import Prophet
from optuna import create_study
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('TASK 1: ACQUIRE AND PREPARE TIME SERIES DATASET')
print('='*80)

np.random.seed(42)
days = 1095
date_range = pd.date_range(start='2021-01-01', periods=days, freq='D')

trend = np.linspace(100, 150, days)
yearly_seasonality = 20 * np.sin(2 * np.pi * np.arange(days) / 365)
weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(days) / 7)

changepoint_idx = int(days * 0.6)
trend[changepoint_idx:] += 15

noise = np.random.normal(0, 5, days)
y_values = (trend + yearly_seasonality + weekly_seasonality) * (1 + noise/200)

for i, date in enumerate(date_range):
    if date.month == 12 and date.day in [24, 25, 26]:
        y_values[i] *= 1.15
    elif date.month == 1 and date.day == 1:
        y_values[i] *= 1.12

df = pd.DataFrame({'ds': date_range, 'y': y_values})

print(f'Dataset Generated: {len(df)} days from {df["ds"].min().date()} to {df["ds"].max().date()}')
print(f'Mean: {df["y"].mean():.2f}, Std: {df["y"].std():.2f}')
print(f'Min: {df["y"].min():.2f}, Max: {df["y"].max():.2f}')

print('\n' + '='*80)
print('TASK 2: ROLLING-ORIGIN CROSS-VALIDATION')
print('='*80)

def rolling_origin_cv(data, train_size=730, test_size=30, step=30):
    folds = []
    n = len(data)
    start = 0
    while start + train_size + test_size <= n:
        train_df = data.iloc[start:start+train_size].copy()
        test_df = data.iloc[start+train_size:start+train_size+test_size].copy()
        folds.append((train_df, test_df))
        start += step
    return folds

cv_folds = rolling_origin_cv(df)
print(f'Created {len(cv_folds)} cross-validation folds')
print(f'Train size: 730 days, Test size: 30 days, Step: 30 days')

print('\n' + '='*80)
print('TASK 3: HYPERPARAMETER OPTIMIZATION')
print('='*80)

def evaluate_prophet(train_df, test_df, params):
    try:
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            n_changepoints=params['n_changepoints'],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(train_df)
        
        future = model.make_future_dataframe(periods=len(test_df))
        forecast = model.predict(future)
        test_forecast = forecast[forecast['ds'] >= test_df['ds'].min()].head(len(test_df))
        
        y_true = test_df['y'].values
        y_pred = test_forecast['yhat'].values
        mae = mean_absolute_error(y_true, y_pred)
        return mae
    except:
        return float('inf')

def objective(trial, folds):
    params = {
        'changepoint_prior_scale': trial.suggest_float('cps', 0.001, 0.5, log=True),
        'seasonality_prior_scale': trial.suggest_float('sps', 0.01, 10, log=True),
        'seasonality_mode': trial.suggest_categorical('sm', ['additive', 'multiplicative']),
        'n_changepoints': trial.suggest_int('nc', 20, 50),
    }
    
    fold_maes = []
    for train, test in folds:
        mae = evaluate_prophet(train, test, params)
        fold_maes.append(mae)
    
    return np.mean(fold_maes)

print('Running Bayesian Optimization (5 trials)...')
sampler = TPESampler(seed=42)
study = create_study(sampler=sampler, direction='minimize')

try:
    study.optimize(lambda trial: objective(trial, cv_folds), n_trials=5, show_progress_bar=False)
    best_params = study.best_params
    best_mae = study.best_value
    print(f'Best CV MAE: {best_mae:.4f}')
    print(f'Optimal changepoint_prior_scale: {best_params["cps"]:.6f}')
    print(f'Optimal seasonality_prior_scale: {best_params["sps"]:.6f}')
    print(f'Optimal seasonality_mode: {best_params["sm"]}')
    print(f'Optimal n_changepoints: {best_params["nc"]}')
    final_params = {
        'changepoint_prior_scale': best_params['cps'],
        'seasonality_prior_scale': best_params['sps'],
        'seasonality_mode': best_params['sm'],
        'n_changepoints': best_params['nc'],
    }
except Exception as e:
    print(f'Using default parameters')
    final_params = {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'multiplicative',
        'n_changepoints': 25,
    }

print('\n' + '='*80)
print('TASK 4: FINAL MODEL TRAINING AND COMPARISON')
print('='*80)

split_idx = int(len(df) * 0.9)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f'Training set: {len(train_df)} days')
print(f'Test set: {len(test_df)} days')

print('\nTraining Optimized Prophet Model...')
prophet_model = Prophet(
    changepoint_prior_scale=final_params['changepoint_prior_scale'],
    seasonality_prior_scale=final_params['seasonality_prior_scale'],
    seasonality_mode=final_params['seasonality_mode'],
    n_changepoints=final_params['n_changepoints'],
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95
)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    prophet_model.fit(train_df)

future = prophet_model.make_future_dataframe(periods=len(test_df))
prophet_forecast = prophet_model.predict(future)
prophet_pred = prophet_forecast[prophet_forecast['ds'] >= test_df['ds'].min()].head(len(test_df))['yhat'].values

print('Training Baseline (Simple Exponential Smoothing)...')
baseline_model = SimpleExpSmoothing(train_df['y'].values, initialization_method='estimated').fit(optimized=True)
baseline_pred = baseline_model.forecast(steps=len(test_df))

y_true = test_df['y'].values

prophet_mae = mean_absolute_error(y_true, prophet_pred)
prophet_rmse = np.sqrt(mean_squared_error(y_true, prophet_pred))
prophet_mape = mean_absolute_percentage_error(y_true, prophet_pred)

baseline_mae = mean_absolute_error(y_true, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
baseline_mape = mean_absolute_percentage_error(y_true, baseline_pred)

print('\n' + '='*80)
print('PERFORMANCE METRICS')
print('='*80)

print(f'\nOptimized Prophet:')
print(f'  MAE:  {prophet_mae:.4f}')
print(f'  RMSE: {prophet_rmse:.4f}')
print(f'  MAPE: {prophet_mape:.4f}%')

print(f'\nBaseline (Simple Exponential Smoothing):')
print(f'  MAE:  {baseline_mae:.4f}')
print(f'  RMSE: {baseline_rmse:.4f}')
print(f'  MAPE: {baseline_mape:.4f}%')

improvement_mae = ((baseline_mae - prophet_mae) / baseline_mae * 100)
print(f'\nImprovement: {improvement_mae:+.2f}%')

print('\n' + '='*80)
print('PROJECT COMPLETE')
print('='*80)

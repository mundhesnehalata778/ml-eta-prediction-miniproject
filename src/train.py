import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from features import engineer_features, get_feature_columns
from ingest import load_raw_data
from validate import run_validation
 
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('ETA-Prediction')
 
def load_features():
    df = load_raw_data()
    df = run_validation(df)
    df = engineer_features(df)
    X = df[get_feature_columns()].fillna(0)
    y = df['log_trip_duration']
    return train_test_split(X, y, test_size=0.2, random_state=42)
 
def evaluate(y_true, y_pred):
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    return {
        'mae_seconds': mean_absolute_error(y_true_orig, y_pred_orig),
        'mae_minutes': mean_absolute_error(y_true_orig, y_pred_orig) / 60,
        'rmse_log':    np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2':          r2_score(y_true, y_pred)
    }
 
def train_model(model, name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param('model_type', name)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        metrics = evaluate(y_test, y_pred)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if name == 'XGBoost':
            mlflow.xgboost.log_model(model, 'model')
        else:
            mlflow.sklearn.log_model(model, 'model')
        joblib.dump(model,  f'{MODELS_DIR}/{name}_model.pkl')
        joblib.dump(scaler, f'{MODELS_DIR}/{name}_scaler.pkl')
        print(f'{name}: MAE={metrics["mae_seconds"]:.1f}s  R2={metrics["r2"]:.4f}')
        return metrics
 
if __name__ == '__main__':
    print('Loading features...')
    X_train, X_test, y_train, y_test = load_features()
    print(f'Train size: {len(X_train):,} rows, Test size: {len(X_test):,} rows')
    results = {}
 
    print('Training LinearRegression...')
    results['LinearRegression'] = train_model(
        LinearRegression(), 'LinearRegression', {'algorithm':'OLS'},
        X_train,X_test,y_train,y_test)
 
    print('Training Ridge...')
    results['Ridge'] = train_model(
        Ridge(alpha=1.0), 'Ridge', {'alpha':1.0},
        X_train,X_test,y_train,y_test)
 
    print('Training GradientBoosting (this is usually the slowest step)...')
    results['GradientBoosting'] = train_model(
        GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=5),
        'GradientBoosting', {'n_estimators':200,'lr':0.1,'depth':5},
        X_train,X_test,y_train,y_test)
 
    print('Training XGBoost...')
    results['XGBoost'] = train_model(
        XGBRegressor(n_estimators=300,learning_rate=0.05,max_depth=6,eval_metric='rmse'),
        'XGBoost', {'n_estimators':300,'lr':0.05,'depth':6},
        X_train,X_test,y_train,y_test)
 
    best = min(results, key=lambda k: results[k]['mae_seconds'])
    print(f'\nBest model: {best}')
    print('Run: mlflow ui --backend-store-uri sqlite:///mlflow.db  to see experiment dashboard')
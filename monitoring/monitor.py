import pandas as pd

import numpy as np

import json, os

from datetime import datetime

from evidently import Report

from evidently.presets import DataDriftPreset
 
LOGS_DIR = 'monitoring/logs'

os.makedirs(LOGS_DIR, exist_ok=True)
 
def log_prediction(features: dict, prediction: float, actual: float = None):

    entry = {'timestamp': datetime.now().isoformat(),

             'features': features, 'prediction': prediction, 'actual': actual}

    with open(f'{LOGS_DIR}/predictions.jsonl', 'a') as f:

        f.write(json.dumps(entry) + '\n')
 
def simulate_drift(df, drift_type='rush_hour'):

    df_drift = df.copy()

    if drift_type == 'rush_hour':

        print('Simulating RUSH HOUR drift: 1.5x trip durations')

        df_drift['trip_duration'] = df_drift['trip_duration'] * 1.5

        df_drift['is_rush_hour']  = 1

        df_drift['hour_of_day']   = 8

    elif drift_type == 'seasonal':

        print('Simulating SEASONAL drift: winter 20% longer trips')

        df_drift['trip_duration'] = df_drift['trip_duration'] * 1.2

        df_drift['month'] = 1

    return df_drift
 
def check_drift_threshold(baseline_mae, current_mae, threshold=0.15):

    degradation = (current_mae - baseline_mae) / baseline_mae

    return {

        'drift_detected':   degradation > threshold,

        'baseline_mae':     round(baseline_mae, 2),

        'current_mae':      round(current_mae, 2),

        'degradation_pct':  round(degradation*100, 1),

        'recommendation':   'RETRAIN' if degradation > threshold else 'MONITOR'

    }
 
def generate_drift_report(reference_df, current_df, feature_cols):

    report = Report([DataDriftPreset()])

    my_eval = report.run(

        reference_data=reference_df[feature_cols],

        current_data=current_df[feature_cols]

    )

    my_eval.save_html('monitoring/drift_report.html')

    print('Drift report saved: monitoring/drift_report.html')
 
if __name__ == '__main__':

    import sys; sys.path.insert(0,'src')

    from ingest import load_raw_data

    from validate import run_validation

    from features import engineer_features, get_feature_columns

    import joblib

    df = load_raw_data()

    df = run_validation(df)

    df = engineer_features(df)

    feature_cols = get_feature_columns()

    model  = joblib.load('models/XGBoost_model.pkl')

    scaler = joblib.load('models/XGBoost_scaler.pkl')

    # Baseline: first 10,000 rows

    ref  = df.head(10000)

    curr = df.iloc[10000:12000]

    ref_pred  = np.expm1(model.predict(scaler.transform(ref[feature_cols].fillna(0))))

    ref_act   = np.expm1(ref['log_trip_duration'])

    base_mae  = float(np.mean(np.abs(ref_pred - ref_act)))

    print(f'Baseline MAE: {base_mae:.1f} seconds')
 
    curr_drifted = simulate_drift(curr, drift_type='rush_hour')

    curr_pred = np.expm1(model.predict(scaler.transform(curr_drifted[feature_cols].fillna(0))))

    curr_act  = np.expm1(curr_drifted['log_trip_duration'])

    curr_mae  = float(np.mean(np.abs(curr_pred - curr_act)))
 
    drift_check = check_drift_threshold(base_mae, curr_mae)

    print(f'Drift check: {drift_check}')
 
    generate_drift_report(ref, curr_drifted, feature_cols) 
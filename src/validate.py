import pandas as pd
import numpy as np

def validate_schema(df):
    required = ['id','vendor_id','pickup_datetime','dropoff_datetime',
                'passenger_count','pickup_longitude','pickup_latitude',
                'dropoff_longitude','dropoff_latitude','trip_duration']
    report = {'total_rows': len(df), 'errors': [], 'warnings': []}
    missing = [c for c in required if c not in df.columns]
    if missing:
        report['errors'].append(f'Missing columns: {missing}')
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            pct = (count / len(df)) * 100
            report['warnings'].append(f'{col}: {count} nulls ({pct:.1f}%)')
    return report

def validate_values(df):
    report = {'errors': [], 'warnings': [], 'rows_removed': 0}
    original_len = len(df)
    # Remove invalid trip durations (must be 0-24 hours)
    df = df[(df['trip_duration'] > 0) & (df['trip_duration'] <= 86400)]
    # Remove trips outside NYC bounding box
    df = df[(df['pickup_longitude'].between(-74.05, -73.75)) &
            (df['pickup_latitude'].between(40.60, 40.90)) &
            (df['dropoff_longitude'].between(-74.05, -73.75)) &
            (df['dropoff_latitude'].between(40.60, 40.90))]
    # Remove invalid passenger counts
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    report['rows_removed'] = original_len - len(df)
    report['rows_remaining'] = len(df)
    return report, df

def run_validation(df):
    print('=== SCHEMA VALIDATION ===')
    schema_report = validate_schema(df)
    print(f'Total rows: {schema_report["total_rows"]:,}')
    for w in schema_report['warnings']:
        print(f'WARNING: {w}')
    print('\n=== VALUE VALIDATION ===')
    value_report, df_clean = validate_values(df)
    print(f'Rows removed : {value_report["rows_removed"]:,}')
    print(f'Rows kept    : {value_report["rows_remaining"]:,}')
    return df_clean

if __name__ == '__main__':
    from ingest import load_raw_data
    df = load_raw_data()
    df_clean = run_validation(df)
    print(df_clean.shape)

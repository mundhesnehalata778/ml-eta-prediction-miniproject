import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    # Calculate straight-line distance in km between two GPS points
    R = 6371
    lat1,lon1,lat2,lon2 = map(np.radians,[lat1,lon1,lat2,lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def engineer_features(df):
    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    # Time features
    df['hour_of_day']  = df['pickup_datetime'].dt.hour
    df['day_of_week']  = df['pickup_datetime'].dt.dayofweek
    df['month']        = df['pickup_datetime'].dt.month
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour_of_day'].apply(
        lambda h: 1 if (7<=h<=9) or (16<=h<=19) else 0)
    df['is_night']     = df['hour_of_day'].apply(
        lambda h: 1 if (h>=22) or (h<=5) else 0)
    # Distance and direction features
    df['distance_km'] = haversine_distance(
        df['pickup_latitude'],  df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude'])
    df['direction'] = np.degrees(np.arctan2(
        df['dropoff_latitude']  - df['pickup_latitude'],
        df['dropoff_longitude'] - df['pickup_longitude']))
    df['pickup_distance_from_center'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'], 40.7580, -73.9855)
    # Log-transform the target variable
    if 'trip_duration' in df.columns:
        df['log_trip_duration'] = np.log1p(df['trip_duration'])
    print(f'Feature engineering done. Total columns: {df.shape[1]}')
    return df

def get_feature_columns():
    return ['passenger_count','distance_km','hour_of_day','day_of_week',
            'month','is_weekend','is_rush_hour','is_night','direction',
            'pickup_distance_from_center','pickup_latitude','pickup_longitude',
            'dropoff_latitude','dropoff_longitude']

if __name__ == '__main__':
    from ingest import load_raw_data, save_processed
    from validate import run_validation
    df = load_raw_data()
    df = run_validation(df)
    df = engineer_features(df)
    save_processed(df)
    print(df[get_feature_columns()].head())

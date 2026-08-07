import pandas as pd
import os

RAW_PATH = 'data/raw/train.csv'
PROCESSED_PATH = 'data/processed/trips_clean.csv'

def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    print(f'Loading data from {path}...')
    df = pd.read_csv(path)
    print(f'Loaded {len(df):,} rows and {len(df.columns)} columns')
    print(f'Columns: {list(df.columns)}')
    return df

def save_processed(df: pd.DataFrame, path: str = PROCESSED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f'Saved {len(df):,} rows to {path}')

if __name__ == '__main__':
    df = load_raw_data()
    print(df.head())
    print(df.dtypes)

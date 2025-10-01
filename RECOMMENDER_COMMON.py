import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

def load_ratings_movielens(path=None):
    if path is None:
        print("No path provided: generating synthetic dataset (small).")
        n_users = 50
        n_items = 100
        data = []
        rng = np.random.default_rng(42)
        for u in range(n_users):
            for _ in range(20):
                i = int(rng.integers(0, n_items))
                r = float(rng.integers(1, 6))
                data.append((u, i, r))
        df = pd.DataFrame(data, columns=['userId', 'itemId', 'rating'])
        return df
    else:
        try:
            df = pd.read_csv(path)
            if {'userId','movieId','rating'}.issubset(df.columns):
                df = df.rename(columns={'movieId':'itemId'})
                df = df[['userId','itemId','rating']]
                return df
            if df.shape[1] == 4 and df.columns.tolist() == ['userId','itemId','rating','timestamp']:
                return df[['userId','itemId','rating']]
            return df
        except Exception as e:
            try:
                df = pd.read_csv(path, sep='\t', names=['userId','itemId','rating','timestamp'])
                return df[['userId','itemId','rating']]
            except Exception:
                raise e

def train_test_split_ratings(df, test_size=0.2, seed=42):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)

def build_user_item_matrix(df):
    users = df['userId'].unique()
    items = df['itemId'].unique()
    user_to_idx = {u:i for i,u in enumerate(users)}
    item_to_idx = {i:j for j,i in enumerate(items)}
    R = np.zeros((len(users), len(items)))
    for _, row in df.iterrows():
        R[user_to_idx[row['userId']], item_to_idx[row['itemId']]] = row['rating']
    return R, user_to_idx, item_to_idx

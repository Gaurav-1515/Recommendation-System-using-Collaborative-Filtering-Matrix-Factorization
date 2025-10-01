import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import random
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
            print("Loaded CSV, expecting columns userId,itemId,rating.")
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

import numpy as np
import pandas as pd
from RECOMMENDER_COMMON import load_ratings_movielens, train_test_split_ratings, build_user_item_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import math

DATA_PATH = None  
df = load_ratings_movielens(DATA_PATH)
train_df, test_df = train_test_split_ratings(df, test_size=0.2)


R_train, user_to_idx, item_to_idx = build_user_item_matrix(train_df)

idx_to_user = {v:k for k,v in user_to_idx.items()}
idx_to_item = {v:k for k,v in item_to_idx.items()}
user_sim = cosine_similarity(R_train)
user_sim = np.nan_to_num(user_sim)
def predict_user_based(u_idx, i_idx, R, sim, k=None):
    
    rated_by = np.where(R[:, i_idx] > 0)[0]
    if len(rated_by) == 0:
        return R[R > 0].mean() if np.any(R>0) else 3.0
    sims = sim[u_idx, rated_by]
    if k:
        top_k_idx = np.argsort(sims)[-k:]
        rated_by = rated_by[top_k_idx]
        sims = sims[top_k_idx]
    numer = np.dot(sims, R[rated_by, i_idx])
    denom = np.sum(np.abs(sims)) + 1e-9
    return numer / denom

preds = []
truths = []
for _, row in test_df.iterrows():
    u = row['userId']
    i = row['itemId']
    r = row['rating']
    if u in user_to_idx and i in item_to_idx:
        p = predict_user_based(user_to_idx[u], item_to_idx[i], R_train, user_sim, k=30)
    else:
        p = train_df['rating'].mean() 
    preds.append(p)
    truths.append(r)

rmse = math.sqrt(mean_squared_error(truths, preds))
print("User-based CF RMSE:", rmse)
from RECOMMENDER_COMMON import load_ratings_movielens, train_test_split_ratings, build_user_item_matrix
from sklearn.metrics.pairwise import cosine_similarity
import math
from sklearn.metrics import mean_squared_error

DATA_PATH = None
df = load_ratings_movielens(DATA_PATH)
train_df, test_df = train_test_split_ratings(df, test_size=0.2)
R_train, user_to_idx, item_to_idx = build_user_item_matrix(train_df)
item_sim = cosine_similarity(R_train.T)
item_sim = np.nan_to_num(item_sim)

def predict_item_based(u_idx, i_idx, R, sim, k=None):
    rated_items = np.where(R[u_idx, :] > 0)[0]
    if len(rated_items) == 0:
        return R[R > 0].mean() if np.any(R>0) else 3.0
    sims = sim[i_idx, rated_items]
    if k:
        top_k_idx = np.argsort(sims)[-k:]
        rated_items = rated_items[top_k_idx]
        sims = sims[top_k_idx]
    numer = np.dot(sims, R[u_idx, rated_items])
    denom = np.sum(np.abs(sims)) + 1e-9
    return numer / denom

preds = []
truths = []
for _, row in test_df.iterrows():
    u = row['userId']; i = row['itemId']; r = row['rating']
    if u in user_to_idx and i in item_to_idx:
        p = predict_item_based(user_to_idx[u], item_to_idx[i], R_train, item_sim, k=30)
    else:
        p = train_df['rating'].mean()
    preds.append(p); truths.append(r)

rmse = math.sqrt(mean_squared_error(truths, preds))
print("Item-based CF RMSE:", rmse)
import numpy as np
import pandas as pd
from RECOMMENDER_COMMON import load_ratings_movielens, train_test_split_ratings
from sklearn.metrics import mean_squared_error
import math
import random

DATA_PATH = None
df = load_ratings_movielens(DATA_PATH)
train_df, test_df = train_test_split_ratings(df, test_size=0.2)
users = train_df['userId'].unique()
items = train_df['itemId'].unique()
u2i = {u:idx for idx,u in enumerate(users)}
i2i = {i:idx for idx,i in enumerate(items)}
train_triplets = [(u2i[r['userId']], i2i[r['itemId']], r['rating']) for _, r in train_df.iterrows() if r['userId'] in u2i and r['itemId'] in i2i]

n_users = len(u2i)
n_items = len(i2i)
K = 20  
mu = train_df['rating'].mean()


P = np.random.normal(scale=0.1, size=(n_users, K))
Q = np.random.normal(scale=0.1, size=(n_items, K))
bu = np.zeros(n_users)
bi = np.zeros(n_items)
lr = 0.005
reg = 0.02
epochs = 20
for epoch in range(epochs):
    random.shuffle(train_triplets)
    for (u,i,r) in train_triplets:
        pred = mu + bu[u] + bi[i] + P[u,:].dot(Q[i,:])
        e = r - pred
        bu[u] += lr * (e - reg * bu[u])
        bi[i] += lr * (e - reg * bi[i])
        P[u,:] += lr * (e * Q[i,:] - reg * P[u,:])
        Q[i,:] += lr * (e * P[u,:] - reg * Q[i,:])
    train_preds = []
    train_truths = []
    for (u,i,r) in train_triplets:
        train_preds.append(mu + bu[u] + bi[i] + P[u,:].dot(Q[i,:]))
        train_truths.append(r)
    train_rmse = math.sqrt(mean_squared_error(train_truths, train_preds))
    print(f"Epoch {epoch+1}/{epochs} train RMSE: {train_rmse:.4f}")

preds = []
truths = []
for _, row in test_df.iterrows():
    u_raw, i_raw, r = row['userId'], row['itemId'], row['rating']
    if u_raw in u2i and i_raw in i2i:
        u = u2i[u_raw]; i = i2i[i_raw]
        p = mu + bu[u] + bi[i] + P[u,:].dot(Q[i,:])
    else:
        p = mu
    preds.append(p); truths.append(r)
print("FunkSVD Test RMSE:", math.sqrt(mean_squared_error(truths, preds)))
import numpy as np
import pandas as pd
from RECOMMENDER_COMMON import load_ratings_movielens, train_test_split_ratings, build_user_item_matrix
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import math
def precision_at_k_for_model(test_df, score_fn, k=10, min_rating=4.0):
    relevant = {}
    for _, row in test_df.iterrows():
        u = row['userId']; i = row['itemId']; r = row['rating']
        if r >= min_rating:
            relevant.setdefault(u, set()).add(i)
    precisions = []
    users = list(relevant.keys())
    for u in users:
        scores = score_fn(u)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        topk = [item for item, s in ranked[:k]]
        rel = relevant.get(u, set())
        hits = sum(1 for it in topk if it in rel)
        precisions.append(hits / k)
    return np.mean(precisions)

def funk_score_fn(user_raw_id):
    if user_raw_id not in u2i:
        return {k: mu for k in i2i.keys()}
    uidx = u2i[user_raw_id]
    scores = {}
    for item_raw, iid in i2i.items():
        scores[item_raw] = mu + bu[uidx] + bi[iid] + P[uidx].dot(Q[iid])
    return scores
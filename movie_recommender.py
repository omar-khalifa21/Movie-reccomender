import argparse
import os
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)


def load_movielens_100k(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expects MovieLens 100K files in data_dir:
      - u.data (user_id, item_id, rating, timestamp) tab-separated
      - u.item (movie_id | title | release | ... ) pipe-separated
    Returns (ratings_df, movies_df)
    """
    ratings_path = os.path.join(data_dir, 'u.data')
    items_path = os.path.join(data_dir, 'u.item')
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Expected {ratings_path}; download MovieLens 100K and place files in {data_dir}")
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Expected {items_path}; download MovieLens 100K and place files in {data_dir}")

    ratings = pd.read_csv(
        ratings_path,
        sep='	',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )

    movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL']
    movies_cols += [f'g{i}' for i in range(19)]  

    movies = pd.read_csv(
        items_path,
        sep='|',
        names=movies_cols,
        encoding='latin-1'
    )[['movie_id', 'title']]

    return ratings, movies


def make_train_test(ratings: pd.DataFrame, test_size_per_user: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-N-out per user (default 5). Users with < N+1 ratings keep 1 in test if possible."""
    ratings = ratings.sort_values('timestamp')
    test_rows = []
    train_rows = []

    for _, grp in ratings.groupby('user_id'):
        if len(grp) <= test_size_per_user:
            if len(grp) == 1:
                train_rows.append(grp.iloc[0])
            else:
                test_rows.append(grp.iloc[-1])
                train_rows.extend(grp.iloc[:-1].to_dict('records'))
            continue
        test_rows.extend(grp.iloc[-test_size_per_user:].to_dict('records'))
        train_rows.extend(grp.iloc[:-test_size_per_user].to_dict('records'))

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, test_df

def build_ui_matrix(ratings: pd.DataFrame) -> Tuple[csr_matrix, Dict[int,int], Dict[int,int], Dict[int,int], Dict[int,int]]:
    users = sorted(ratings['user_id'].unique())
    items = sorted(ratings['item_id'].unique())
    uid_to_ix = {u:i for i,u in enumerate(users)}
    iid_to_ix = {m:i for i,m in enumerate(items)}
    ix_to_uid = {i:u for u,i in uid_to_ix.items()}
    ix_to_iid = {i:m for m,i in iid_to_ix.items()}

    row = ratings['user_id'].map(uid_to_ix)
    col = ratings['item_id'].map(iid_to_ix)
    data = ratings['rating'].astype(float)
    mat = csr_matrix((data, (row, col)), shape=(len(users), len(items)))
    return mat, uid_to_ix, iid_to_ix, ix_to_uid, ix_to_iid


def normalize_users(R: csr_matrix) -> Tuple[csr_matrix, np.ndarray]:
    R = R.tocsr()
    means = np.zeros(R.shape[0])
    R_norm = R.copy().astype(float)
    for i in range(R.shape[0]):
        start, end = R.indptr[i], R.indptr[i+1]
        if end > start:
            row_data = R.data[start:end]
            mu = row_data.mean()
            means[i] = mu
            R_norm.data[start:end] = row_data - mu
    return R_norm, means


def _prune_topk_lil(sim_lil, k: int, axis_name: str):
    """Keep only top-k nonzeros per row for LIL matrix (works for user-user or item-item)."""
    n = sim_lil.shape[0]
    for i in range(n):
        row_idx_list = sim_lil.rows[i]     
        row_val_list = sim_lil.data[i]     
        if len(row_val_list) > k:
            vals_np = np.array(row_val_list)
            topk_idx = np.argpartition(vals_np, -k)[-k:]
            keep_set = {row_idx_list[idx] for idx in topk_idx}
            new_rows, new_vals = [], []
            for r, v in zip(row_idx_list, row_val_list):
                if r in keep_set:
                    new_rows.append(r)
                    new_vals.append(v)
            sim_lil.rows[i] = new_rows
            sim_lil.data[i] = new_vals


def predict_user_based(train_mat: csr_matrix, k: int = 40) -> csr_matrix:
    R_norm, user_means = normalize_users(train_mat)
    sim = cosine_similarity(R_norm, dense_output=False)  
    sim = sim.tolil()
    _prune_topk_lil(sim, k, axis_name='user')
    sim = sim.tocsr()

    numer = (sim @ R_norm).toarray()  
    denom = np.abs(sim).sum(axis=1).A1 + 1e-8  
    pred = numer / denom[:, None]
    pred += user_means[:, None]
    return csr_matrix(pred)


def predict_item_based(train_mat: csr_matrix, k: int = 50) -> csr_matrix:
    R = train_mat.tocsc()
    sim = cosine_similarity(R.T, dense_output=False)  # item-item (sparse)
    sim = sim.tolil()
    _prune_topk_lil(sim, k, axis_name='item')
    sim = sim.tocsc()

    numer = (train_mat @ sim).toarray()  # (n_users x n_items)
    denom = np.abs(sim).sum(axis=0).A1 + 1e-8  
    pred = numer / denom[None, :]
    return csr_matrix(pred)

# SVD (Matrix Factorization)

def predict_svd(train_mat: csr_matrix, rank: int = 50) -> csr_matrix:
    R = train_mat.copy().astype(float)
    R_norm, user_means = normalize_users(R)
    # truncated SVD on sparse
    rank = min(rank, min(R_norm.shape) - 1) if min(R_norm.shape) > 1 else 1
    U, s, Vt = svds(R_norm.asfptype(), k=rank)
    idx = np.argsort(-s)
    U, s, Vt = U[:, idx], s[idx], Vt[idx, :]
    S = np.diag(s)
    R_hat = U @ S @ Vt
    R_hat += user_means[:, None]
    return csr_matrix(R_hat)

# Recommend & Evaluate (Precision@K)

def get_user_known_items(mat: csr_matrix, uix: int) -> set:
    start, end = mat.indptr[uix], mat.indptr[uix+1]
    return set(mat.indices[start:end])


def recommend_top_n(pred: csr_matrix, train: csr_matrix, uix: int, top_n: int = 10) -> List[int]:
    seen = get_user_known_items(train, uix)
    scores = pred.getrow(uix).toarray().ravel()
    if len(scores) == 0:
        return []
    if seen:
        scores[list(seen)] = -np.inf
    top_n = min(top_n, np.count_nonzero(np.isfinite(scores)))
    if top_n <= 0:
        return []
    top_idx = np.argpartition(scores, -top_n)[-top_n:]
    return top_idx[np.argsort(-scores[top_idx])].tolist()


def precision_at_k(pred: csr_matrix, train: csr_matrix, test_df: pd.DataFrame, uid_to_ix: Dict[int,int], iid_to_ix: Dict[int,int], k: int = 10, threshold: float = 4.0) -> float:
    test_pos: Dict[int, set] = {}
    for _, row in test_df.iterrows():
        if row['rating'] >= threshold and row['user_id'] in uid_to_ix and row['item_id'] in iid_to_ix:
            uix = uid_to_ix[row['user_id']]
            iix = iid_to_ix[row['item_id']]
            test_pos.setdefault(uix, set()).add(iix)

    if not test_pos:
        return 0.0

    precisions = []
    for uix in test_pos.keys():
        recs = recommend_top_n(pred, train, uix, top_n=k)
        if not recs:
            continue
        hits = len(set(recs) & test_pos[uix])
        precisions.append(hits / float(k))

    return float(np.mean(precisions)) if precisions else 0.0

# Runner

def run(data_dir: str, method: str, k: int, rank: int, top_n: int, prec_k: int, threshold: float):
    ratings, movies = load_movielens_100k(data_dir)
    train_df, test_df = make_train_test(ratings, test_size_per_user=5)

    train_mat, uid_to_ix, iid_to_ix, ix_to_uid, ix_to_iid = build_ui_matrix(train_df)

    if method == 'usercf':
        pred = predict_user_based(train_mat, k=k)
    elif method == 'itemcf':
        pred = predict_item_based(train_mat, k=k)
    elif method == 'svd':
        pred = predict_svd(train_mat, rank=rank)
    elif method == 'compare':
        pred_user = predict_user_based(train_mat, k=k)
        pred_item = predict_item_based(train_mat, k=k)
        pred_svd = predict_svd(train_mat, rank=rank)

        pu = precision_at_k(pred_user, train_mat, test_df, uid_to_ix, iid_to_ix, k=prec_k, threshold=threshold)
        pi = precision_at_k(pred_item, train_mat, test_df, uid_to_ix, iid_to_ix, k=prec_k, threshold=threshold)
        ps = precision_at_k(pred_svd, train_mat, test_df, uid_to_ix, iid_to_ix, k=prec_k, threshold=threshold)

        print(f"Precision@{prec_k} (UserCF, k={k}): {pu:.4f}")
        print(f"Precision@{prec_k} (ItemCF, k={k}): {pi:.4f}")
        print(f"Precision@{prec_k} (SVD, rank={rank}): {ps:.4f}")

        sample_uid = list(uid_to_ix.keys())[0]
        uix = uid_to_ix[sample_uid]

        def titles_from_indices(indices: List[int]) -> List[str]:
            inv = {v: k for k, v in iid_to_ix.items()}
            ids = [inv[i] for i in indices]
            return movies[movies['movie_id'].isin(ids)].set_index('movie_id').loc[ids]['title'].tolist()

        for name, P in [('UserCF', pred_user), ('ItemCF', pred_item), ('SVD', pred_svd)]:
            rec_idx = recommend_top_n(P, train_mat, uix, top_n=top_n)
            rec_titles = titles_from_indices(rec_idx)
            print(f"Top-{top_n} for user {sample_uid} with {name}:")
            for t in rec_titles:
                print(" -", t)
        return
    else:
        raise ValueError("method must be one of {usercf, itemcf, svd, compare}")

    p = precision_at_k(pred, train_mat, test_df, uid_to_ix, iid_to_ix, k=prec_k, threshold=threshold)
    print(f"Precision@{prec_k} ({method}): {p:.4f}")

    sample_uid = list(uid_to_ix.keys())[0]
    uix = uid_to_ix[sample_uid]
    inv = {v: k for k, v in iid_to_ix.items()}
    rec_idx = recommend_top_n(pred, train_mat, uix, top_n=top_n)
    ids = [inv[i] for i in rec_idx]
    titles = movies[movies['movie_id'].isin(ids)].set_index('movie_id').loc[ids]['title'].tolist()

    print(f"Top-{top_n} recommendations for user {sample_uid}:")
    for t in titles:
        print(" -", t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MovieLens 100K Recommender (UserCF, ItemCF, SVD)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to MovieLens 100K folder (contains u.data, u.item)')
    parser.add_argument('--method', type=str, default='usercf', choices=['usercf', 'itemcf', 'svd', 'compare'])
    parser.add_argument('--k', type=int, default=40, help='Neighborhood size for CF')
    parser.add_argument('--rank', type=int, default=50, help='Rank for SVD')
    parser.add_argument('--top_n', type=int, default=10, help='Top-N recommendations to show')
    parser.add_argument('--prec_k', type=int, default=10, help='K for Precision@K evaluation')
    parser.add_argument('--threshold', type=float, default=4.0, help='Positive rating threshold for evaluation')

    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        method=args.method,
        k=args.k,
        rank=args.rank,
        top_n=args.top_n,
        prec_k=args.prec_k,
        threshold=args.threshold,
    )
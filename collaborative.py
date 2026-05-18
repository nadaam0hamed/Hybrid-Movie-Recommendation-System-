# =========================================================
# COLLABORATIVE FILTERING MODULE (محسّن مع SVD الحقيقي والتطبيع)
# =========================================================

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from data_processing import preprocess_ratings


def split_ratings(ratings, test_size=0.2, random_state=42):
    ratings = preprocess_ratings(ratings)
    train_ratings, test_ratings = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=ratings['userId'] if ratings['userId'].nunique() > 1 else None,
    )
    return train_ratings.reset_index(drop=True), test_ratings.reset_index(drop=True)


def build_collaborative_model(train_ratings, n_components=50, random_state=42):
    """
    Build collaborative filtering model from a training split only.
    """
    train_ratings = preprocess_ratings(train_ratings)

    user_counts = train_ratings['userId'].value_counts()
    movie_counts = train_ratings['movieId'].value_counts()

    min_user_ratings = 5
    min_movie_ratings = 3

    filtered_train = train_ratings[
        (train_ratings['userId'].isin(user_counts[user_counts >= min_user_ratings].index)) &
        (train_ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_ratings].index))
    ].reset_index(drop=True)

    user_movie_matrix = filtered_train.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    )

    user_ids = user_movie_matrix.index
    movie_ids = user_movie_matrix.columns

    sparse_matrix = csr_matrix(user_movie_matrix.fillna(0).values)

    n_components = min(n_components, min(sparse_matrix.shape) - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=random_state, n_iter=100)
    matrix_svd = svd.fit_transform(sparse_matrix)
    reconstructed_matrix = np.dot(matrix_svd, svd.components_)

    predicted_ratings = pd.DataFrame(
        reconstructed_matrix,
        index=user_ids,
        columns=movie_ids
    )

    predicted_ratings = predicted_ratings.clip(lower=0.5, upper=5.0)
    return predicted_ratings


def evaluate_collaborative(predicted_ratings, train_ratings, test_ratings, top_n=10, rating_threshold=4.0, fallback=3.0):
    train_ratings = preprocess_ratings(train_ratings)
    test_ratings = preprocess_ratings(test_ratings)

    def get_prediction(row):
        try:
            return float(predicted_ratings.loc[row.userId, row.movieId])
        except (KeyError, TypeError):
            return fallback

    test_ratings = test_ratings.copy()
    test_ratings['predicted'] = test_ratings.apply(get_prediction, axis=1)
    test_ratings = test_ratings.dropna(subset=['rating', 'predicted'])

    actual = test_ratings['rating'].values
    predicted = np.clip(test_ratings['predicted'].values, 0.5, 5.0)

    rmse = float(np.sqrt(mean_squared_error(actual, predicted))) if len(actual) else 0.0
    mae = float(mean_absolute_error(actual, predicted)) if len(actual) else 0.0

    watched_train = train_ratings.groupby('userId')['movieId'].apply(set).to_dict()
    relevant_by_user = test_ratings[test_ratings['rating'] >= rating_threshold].groupby('userId')['movieId'].apply(set).to_dict()

    precision_scores = []
    recall_scores = []
    hit_rates = []

    for user_id, relevant in relevant_by_user.items():
        if user_id not in predicted_ratings.index:
            continue

        watched_movies = watched_train.get(user_id, set())
        user_scores = predicted_ratings.loc[user_id].drop(labels=watched_movies, errors='ignore')
        rec_ids = user_scores.sort_values(ascending=False).head(top_n).index.tolist()

        if not rec_ids:
            continue

        hits = set(rec_ids) & relevant
        precision_scores.append(len(hits) / top_n)
        recall_scores.append(len(hits) / len(relevant))
        hit_rates.append(1.0 if hits else 0.0)

    evaluated_users = len(precision_scores)
    if evaluated_users == 0:
        precision_at_k = recall_at_k = hit_rate = 0.0
    else:
        precision_at_k = float(np.mean(precision_scores))
        recall_at_k = float(np.mean(recall_scores))
        hit_rate = float(np.mean(hit_rates))

    return {
        'rmse': rmse,
        'mae': mae,
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'hit_rate': hit_rate,
        'evaluated_users': evaluated_users,
    }


def predict_rating(predicted_ratings, user_id, movie_id, fallback=3.0):
    try:
        rating = float(predicted_ratings.loc[user_id, movie_id])
        # تأكد من أن التقييم ضمن النطاق الصحيح
        return np.clip(rating, 0.5, 5.0)
    except (KeyError, TypeError):
        return float(fallback)


def recommend_movies_collaborative(user_id, movies, ratings, predicted_ratings, top_n=10, fallback=3.0):
    ratings = preprocess_ratings(ratings)
    
    if user_id not in predicted_ratings.index:
        return pd.DataFrame()  # إرجع dataframe فارغ إذا كان المستخدم غير موجود
    
    watched_movies = set(ratings[ratings['userId'] == user_id]['movieId'].unique())

    candidate_movies = movies[~movies['movieId'].isin(watched_movies)].copy()
    candidate_movies['collab_score'] = candidate_movies['movieId'].apply(
        lambda movie_id: predict_rating(predicted_ratings, user_id, movie_id, fallback=fallback)
    )

    candidate_movies = candidate_movies.sort_values('collab_score', ascending=False).head(top_n)
    return candidate_movies[['movieId', 'title', 'genres', 'collab_score']]

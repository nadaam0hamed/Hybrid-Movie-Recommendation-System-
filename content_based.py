# =========================================================
# CONTENT-BASED MOVIE RECOMMENDATION MODULE
# =========================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data_processing import preprocess_movies


def build_content_model(movies: pd.DataFrame, tags: pd.DataFrame):
    movies_tags = preprocess_movies(movies, tags)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_tags['combined_features'])

    # Use sparse matrix multiplication to avoid building a huge dense similarity matrix
    cosine_sim = tfidf_matrix * tfidf_matrix.T
    indices = pd.Series(movies_tags.index, index=movies_tags['title']).drop_duplicates()

    return movies_tags, tfidf_matrix, cosine_sim, indices


def recommend_movies(title: str, movies_tags: pd.DataFrame, cosine_sim, indices, top_n: int = 10):
    if title not in indices:
        raise ValueError(f"Movie title not found: {title}")

    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx].toarray().ravel()))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1: top_n + 1]

    movie_indices = [i[0] for i in similarity_scores]
    return movies_tags.iloc[movie_indices][['movieId', 'title', 'genres']]


def evaluate_content_model(
    movies: pd.DataFrame,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    movies_tags: pd.DataFrame,
    cosine_sim,
    indices,
    top_n: int = 10,
    rating_threshold: float = 4.0,
):
    train_ratings = train_ratings.copy()
    test_ratings = test_ratings.copy()

    train_favorites = (
        train_ratings.sort_values(['userId', 'rating', 'movieId'], ascending=[True, False, True])
        .groupby('userId', as_index=False)
        .first()
    )

    watched_train = train_ratings.groupby('userId')['movieId'].apply(set).to_dict()
    relevant_by_user = test_ratings[test_ratings['rating'] >= rating_threshold].groupby('userId')['movieId'].apply(set).to_dict()

    precision_scores = []
    recall_scores = []
    hit_rates = []

    for user_id, relevant in relevant_by_user.items():
        if user_id not in watched_train or len(relevant) == 0:
            continue

        favorite_row = train_favorites[train_favorites['userId'] == user_id]
        if favorite_row.empty:
            continue

        favorite_movie_id = int(favorite_row['movieId'].iloc[0])
        title_row = movies[movies['movieId'] == favorite_movie_id]
        if title_row.empty:
            continue

        favorite_title = title_row['title'].iloc[0]
        try:
            recs = recommend_movies(favorite_title, movies_tags, cosine_sim, indices, top_n=top_n * 2)
        except ValueError:
            continue

        rec_ids = [mid for mid in recs['movieId'].tolist() if mid not in watched_train[user_id]]
        if not rec_ids:
            continue

        rec_ids = rec_ids[:top_n]
        hits = set(rec_ids) & relevant

        precision_scores.append(len(hits) / top_n)
        recall_scores.append(len(hits) / len(relevant))
        hit_rates.append(1.0 if hits else 0.0)

    evaluated_users = len(precision_scores)
    if evaluated_users == 0:
        return {
            'precision_at_k': 0.0,
            'recall_at_k': 0.0,
            'hit_rate': 0.0,
            'evaluated_users': 0,
        }

    return {
        'precision_at_k': float(np.mean(precision_scores)),
        'recall_at_k': float(np.mean(recall_scores)),
        'hit_rate': float(np.mean(hit_rates)),
        'evaluated_users': evaluated_users,
    }

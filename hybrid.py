import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from collaborative import predict_rating


def hybrid_recommendations(
    user_id,
    favorite_title,
    movies_tags,
    cosine_sim,
    indices,
    ratings,
    predicted_ratings,
    global_mean,
    top_n=10,
    weight_content=0.4,
    weight_collab=0.6
):
    """
    التوصيات الهجينة المحسّنة مع Normalize Scores الأفضل.
    """
    if favorite_title not in indices:
        raise ValueError(f"Movie title not found: {favorite_title}")

    idx = indices[favorite_title]
    # cosine_sim is a sparse matrix; extract the dense row for the chosen movie
    try:
        sim_row = cosine_sim[idx].toarray().ravel()
    except Exception:
        # Fallback if cosine_sim is dense already
        sim_row = np.asarray(cosine_sim[idx]).ravel()

    similarity_scores = list(enumerate(sim_row))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    hybrid_df = pd.DataFrame(similarity_scores, columns=['movie_index', 'content_score'])
    hybrid_df['movieId'] = movies_tags.loc[hybrid_df['movie_index'], 'movieId'].values
    hybrid_df['title'] = movies_tags.loc[hybrid_df['movie_index'], 'title'].values
    hybrid_df['genres'] = movies_tags.loc[hybrid_df['movie_index'], 'genres'].values

    hybrid_df['collab_score'] = hybrid_df['movieId'].apply(
        lambda mid: predict_rating(predicted_ratings, user_id, mid, fallback=global_mean)
    )

    watched_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()
    hybrid_df = hybrid_df[~hybrid_df['movieId'].isin(watched_movies)].copy()

    # If no hybrid candidates remain, fall back to popular / collaborative-only recommendations
    if hybrid_df.empty:
        # Compute simple popularity signals from ratings
        pop = (
            ratings.groupby('movieId')['rating']
            .agg(pop_count='size', pop_avg='mean')
            .reset_index()
        )

        # Merge with movie metadata
        fallback = movies_tags.merge(pop, on='movieId', how='left')
        fallback['pop_count'] = fallback['pop_count'].fillna(0)
        fallback['pop_avg'] = fallback['pop_avg'].fillna(train_ratings['rating'].mean() if 'train_ratings' in locals() else ratings['rating'].mean())

        # Exclude already watched
        fallback = fallback[~fallback['movieId'].isin(watched_movies)].copy()
        if fallback.empty:
            return pd.DataFrame(
                columns=['movieId', 'title', 'genres', 'content_score', 'collab_score', 'final_score']
            )

        # Compute collaborative score where possible
        fallback['collab_score'] = fallback['movieId'].apply(
            lambda mid: predict_rating(predicted_ratings, user_id, mid, fallback=global_mean)
        )

        # Use popularity normalized as content proxy when content unavailable
        pop_scaler = MinMaxScaler(feature_range=(0, 1))
        fallback['pop_norm'] = pop_scaler.fit_transform(fallback[['pop_count']])

        # Compose final score from popularity and collaborative prediction
        fallback['content_score'] = 0.0
        fallback['final_score'] = (
            weight_content * fallback['content_score']
            + weight_collab * fallback['collab_score']
            + (1.0 - (weight_content + weight_collab)) * fallback['pop_norm']
        )

        # Scale final to 0-10 for consistent UI
        final_scaler = MinMaxScaler(feature_range=(0, 10))
        fallback['final_score'] = final_scaler.fit_transform(fallback[['final_score']])

        fallback = fallback.sort_values(by='final_score', ascending=False).head(top_n)

        return fallback[['movieId', 'title', 'genres', 'content_score', 'collab_score', 'final_score']]

    # ===== تحسين التطبيع =====
    # استخدم MinMaxScaler للنتائج بين 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    hybrid_df[['content_score', 'collab_score']] = scaler.fit_transform(
        hybrid_df[['content_score', 'collab_score']]
    )

    # استخدم المتوسط الحسابي المرجح لتجميع النقاط النهائية
    hybrid_df['final_score'] = (
        weight_content * hybrid_df['content_score']
        + weight_collab * hybrid_df['collab_score']
    )
    
    # تطبيع درجات النهاية
    final_scaler = MinMaxScaler(feature_range=(0, 10))
    hybrid_df['final_score'] = final_scaler.fit_transform(hybrid_df[['final_score']])

    hybrid_df = hybrid_df.sort_values(by='final_score', ascending=False).head(top_n)

    return hybrid_df[
        ['movieId', 'title', 'genres', 'content_score', 'collab_score', 'final_score']
    ]


def evaluate_hybrid_model(
    movies: pd.DataFrame,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    movies_tags: pd.DataFrame,
    cosine_sim,
    indices,
    predicted_ratings: pd.DataFrame,
    global_mean: float,
    top_n: int = 10,
    weight_content: float = 0.4,
    weight_collab: float = 0.6,
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
            recs = hybrid_recommendations(
                user_id=user_id,
                favorite_title=favorite_title,
                movies_tags=movies_tags,
                cosine_sim=cosine_sim,
                indices=indices,
                ratings=train_ratings,
                predicted_ratings=predicted_ratings,
                global_mean=global_mean,
                top_n=top_n,
                weight_content=weight_content,
                weight_collab=weight_collab,
            )
        except Exception:
            continue

        if recs.empty:
            continue

        rec_ids = recs['movieId'].tolist()
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

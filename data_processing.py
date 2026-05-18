import pandas as pd


def load_movielens_data(movies_path='movies.csv', ratings_path='ratings.csv', tags_path='tags.csv'):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)
    return movies, ratings, tags


def preprocess_movies(movies, tags):
    movies = movies.copy()
    tags = tags.copy()

    movies.drop_duplicates(inplace=True)
    tags.drop_duplicates(inplace=True)

    movies['genres'] = movies['genres'].fillna('')

    tags = tags[tags['movieId'].isin(movies['movieId'])]
    tags_grouped = tags.groupby('movieId')['tag'] \
                       .apply(lambda values: ' '.join(values.astype(str))) \
                       .reset_index()

    movies_tags = pd.merge(
        movies,
        tags_grouped,
        on='movieId',
        how='left'
    )

    movies_tags['tag'] = movies_tags['tag'].fillna('')
    movies_tags['genres'] = movies_tags['genres'].str.replace('|', ' ', regex=False)

    movies_tags['combined_features'] = (
        movies_tags['title'].fillna('') + ' ' +
        movies_tags['genres'] + ' ' +
        movies_tags['tag']
    )

    movies_tags.reset_index(drop=True, inplace=True)
    return movies_tags


def preprocess_ratings(ratings):
    ratings = ratings.copy()
    ratings.drop_duplicates(inplace=True)
    ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])

    ratings['userId'] = ratings['userId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)

    return ratings

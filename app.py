# streamlit run app.py
import streamlit as st

from data_processing import load_movielens_data
from content_based import build_content_model, evaluate_content_model
from collaborative import split_ratings, build_collaborative_model, evaluate_collaborative
from hybrid import hybrid_recommendations, evaluate_hybrid_model

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Theme Toggle in Session
# =========================

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# =========================
# Style and Animated Background
# =========================
st.markdown(
    """
    <style>
    :root {
        --primary-color: #cc0066;
        --secondary-color: #1f1f1f;
        --light-bg: #fffbfe;
        --dark-bg: #0a0a0a;
        --light-text: #333;
        --dark-text: #e0e0e0;
        --card-light: rgba(255, 255, 255, 0.95);
        --card-dark: rgba(25, 25, 25, 0.95);
    }

    .stApp {
        background: linear-gradient(-45deg, #ffe6f2, #ffcce6, #ffe6f9);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        overflow: hidden;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .emoji {
        position: fixed;
        font-size: 24px;
        animation: floatUp 12s linear infinite;
        opacity: 0.65;
        z-index: 0;
        pointer-events: none;
    }

    @keyframes floatUp {
        0% {
            transform: translateY(100vh) rotate(0deg);
        }
        100% {
            transform: translateY(-10vh) rotate(360deg);
        }
    }

    .main {
        position: relative;
        z-index: 1;
    }

    /* Movie Card with Hover Animation */
    .movie-card {
        background: var(--card-light);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        position: relative;
        z-index: 1;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left: 4px solid var(--primary-color);
        cursor: pointer;
    }

    .movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 24px rgba(204, 0, 102, 0.15);
        border-left: 4px solid #ff0080;
    }

    /* Movie Poster */
    .movie-poster {
        border-radius: 10px;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    .movie-poster:hover {
        transform: scale(1.05);
    }

    /* Trending Section */
    .trending-container {
        display: flex;
        gap: 16px;
        overflow-x: auto;
        padding: 12px 0;
    }

    .trending-item {
        flex: 0 0 180px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .trending-item:hover {
        transform: translateY(-12px);
    }

    /* Search Box */
    .search-box {
        position: relative;
        margin-bottom: 20px;
    }

    .search-input {
        width: 100%;
        padding: 12px 16px;
        border: 2px solid var(--primary-color);
        border-radius: 10px;
        font-size: 16px;
        transition: all 0.3s ease;
    }

    .search-input:focus {
        outline: none;
        box-shadow: 0 0 12px rgba(204, 0, 102, 0.3);
        border-color: #ff0080;
    }

    /* Button Animation */
    .recommendation-btn {
        background: linear-gradient(135deg, var(--primary-color), #ff0080);
        color: white;
        padding: 12px 32px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(204, 0, 102, 0.3);
    }

    .recommendation-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(204, 0, 102, 0.4);
    }

    /* Score Badge */
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--primary-color), #ff0080);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        margin-top: 8px;
    }

    /* Title Styling */
    .page-title {
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(135deg, var(--primary-color), #ff0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
    }

    </style>

    <div class="emoji" style="left:10%;">🎬</div>
    <div class="emoji" style="left:20%; animation-delay:2s;">🍿</div>
    <div class="emoji" style="left:35%; animation-delay:4s;">💖</div>
    <div class="emoji" style="left:50%; animation-delay:1s;">✨</div>
    <div class="emoji" style="left:65%; animation-delay:3s;">🎥</div>
    <div class="emoji" style="left:80%; animation-delay:5s;">💫</div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Data Loading and Model Building
# =========================


@st.cache_data(show_spinner=False)
def load_data():
    return load_movielens_data()


def safe_metric(metrics, key, fallback=0.0, alt_key=None):
    if key in metrics:
        return metrics[key]
    if alt_key and alt_key in metrics:
        return metrics[alt_key]
    return fallback


@st.cache_resource
def load_models():
    movies, ratings, tags = load_data()
    movies_tags, _, cosine_sim, indices = build_content_model(
        movies, tags)

    train_ratings, test_ratings = split_ratings(ratings)
    predicted_ratings = build_collaborative_model(train_ratings)

    eval_k = 10
    collab_metrics = evaluate_collaborative(
        predicted_ratings,
        train_ratings,
        test_ratings,
        top_n=eval_k,
    )
    content_metrics = evaluate_content_model(
        movies,
        train_ratings,
        test_ratings,
        movies_tags,
        cosine_sim,
        indices,
        top_n=eval_k,
    )
    hybrid_metrics = evaluate_hybrid_model(
        movies=movies,
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        movies_tags=movies_tags,
        cosine_sim=cosine_sim,
        indices=indices,
        predicted_ratings=predicted_ratings,
        global_mean=float(train_ratings['rating'].mean()),
        top_n=eval_k,
    )
    global_mean = float(train_ratings['rating'].mean())

    return (
        movies,
        ratings,
        movies_tags,
        cosine_sim,
        indices,
        predicted_ratings,
        global_mean,
        collab_metrics,
        content_metrics,
        hybrid_metrics,
    )


movies, ratings, movies_tags, cosine_sim, indices, predicted_ratings, global_mean, collab_metrics, content_metrics, hybrid_metrics = load_models()

# =========================
# Sidebar Controls
# =========================

st.sidebar.title("Settings")
st.sidebar.write("Adjust hybrid model weights and evaluation.")

weight_content = st.sidebar.slider(
    "Content-Based Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
)

# Add hybrid weight slider to allow three-way blending
weight_hybrid = st.sidebar.slider(
    "Hybrid Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
)

# Add collaborative weight slider (user requested)
weight_collab_user = st.sidebar.slider(
    "Collaborative Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
)

# Normalize weights if the user sets them such that the sum exceeds 1.0
# Normalize the three sliders so their sum is 1.0 (unless all zero)
raw_sum = weight_content + weight_hybrid + weight_collab_user
if raw_sum == 0:
    # default distribution if user zeros everything
    weight_content_norm = 0.4
    weight_hybrid_norm = 0.2
    weight_collab_norm = 0.4
else:
    weight_content_norm = round(weight_content / raw_sum, 3)
    weight_hybrid_norm = round(weight_hybrid / raw_sum, 3)
    weight_collab_norm = round(weight_collab_user / raw_sum, 3)

# For the algorithm, we use content and collaborative normalized weights
weight_content = weight_content_norm
weight_collab = weight_collab_norm

st.sidebar.markdown(f"**Weights (normalized):** Content={weight_content:.2f}, Hybrid={weight_hybrid_norm:.2f}, Collaborative={weight_collab:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Evaluation K:** {10}")
st.sidebar.subheader("Collaborative Evaluation")
st.sidebar.write(
    f"RMSE: {safe_metric(collab_metrics, 'rmse'):.3f}\n"
    f"MAE: {safe_metric(collab_metrics, 'mae'):.3f}\n"
    f"Precision@10: {safe_metric(collab_metrics, 'precision_at_k', alt_key='precision'):.3f}\n"
    f"Recall@10: {safe_metric(collab_metrics, 'recall_at_k', alt_key='recall'):.3f}\n"
    f"Hit Rate: {safe_metric(collab_metrics, 'hit_rate'):.3f}\n"
    f"Evaluated users: {int(safe_metric(collab_metrics, 'evaluated_users'))}"
)
st.sidebar.markdown("---")
st.sidebar.subheader("Content-Based Evaluation")
st.sidebar.write(
    f"Precision@10: {safe_metric(content_metrics, 'precision_at_k', alt_key='precision'):.3f}\n"
    f"Recall@10: {safe_metric(content_metrics, 'recall_at_k', alt_key='recall'):.3f}\n"
    f"Hit Rate: {safe_metric(content_metrics, 'hit_rate'):.3f}\n"
    f"Evaluated users: {int(safe_metric(content_metrics, 'evaluated_users'))}"
)
st.sidebar.markdown("---")
st.sidebar.subheader("Hybrid Evaluation")
st.sidebar.write(
    f"Precision@10: {safe_metric(hybrid_metrics, 'precision_at_k', alt_key='precision'):.3f}\n"
    f"Recall@10: {safe_metric(hybrid_metrics, 'recall_at_k', alt_key='recall'):.3f}\n"
    f"Hit Rate: {safe_metric(hybrid_metrics, 'hit_rate'):.3f}\n"
    f"Evaluated users: {int(safe_metric(hybrid_metrics, 'evaluated_users'))}"
)

# =========================
# Page Header
# =========================

st.title("🎬 Hybrid Movie Recommendation System")
st.markdown(
    "This app combines content-based similarity and collaborative filtering to recommend movies based on user preferences."
)

# =========================
# Recommendation Inputs
# =========================

user_id = st.number_input(
    "Select User ID",
    min_value=int(ratings['userId'].min()),
    max_value=int(ratings['userId'].max()),
    value=int(ratings['userId'].min()),
    step=1,
)

default_movie_index = movies_tags[movies_tags['title'] == 'Toy Story (1995)'].index
favorite_movie = st.selectbox(
    "Choose a favorite movie for content-based guidance",
    movies_tags['title'].tolist(),
    index=int(default_movie_index[0]) if len(default_movie_index) > 0 else 0
)

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=20,
    value=1,
)

# =========================
# Recommendation Results
# =========================

if st.button("Recommend Movies"):
    try:
        results = hybrid_recommendations(
            user_id=user_id,
            favorite_title=favorite_movie,
            movies_tags=movies_tags,
            cosine_sim=cosine_sim,
            indices=indices,
            ratings=ratings,
            predicted_ratings=predicted_ratings,
            global_mean=global_mean,
            top_n=num_recommendations,
            weight_content=weight_content,
            weight_collab=weight_collab,
        )

        if results.empty:
            st.warning("No hybrid recommendations available for this user and seed movie.")
        else:
            st.subheader("Recommended Movies")
            for _, row in results.iterrows():
                st.markdown(
                    f"""
                    <div class='movie-card'>
                        <strong>{row['title']}</strong>
                        <div>Genres: {row['genres']}</div>
                    <div>Content score: {row['content_score']:.3f}</div>
                    <div>Collaborative score: {row['collab_score']:.3f}</div>
                    <div><strong>Hybrid score: {row['final_score']:.3f}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as error:
        st.error(f"Unable to generate recommendations: {error}")

# =========================
# Dataset summary
# =========================

with st.expander("Dataset summary"):
    st.write(f"Movies: {movies.shape[0]}")
    st.write(f"User ratings: {ratings.shape[0]}")
    st.write(f"Unique users: {ratings['userId'].nunique()}")
    st.write(f"Unique movies: {ratings['movieId'].nunique()}")

with st.expander("Usage notes"):
    st.write(
        "Use the slider to tune content vs collaborative weight. \n"
        "The sidebar shows collaborative evaluation metrics using RMSE, MAE, Precision, Recall, and F1-score.")

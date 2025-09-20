import streamlit as st
import requests
import time
import numpy as np
import pandas as pd

from movie_recommender import (
    load_movielens_100k,
    make_train_test,
    build_ui_matrix,
    predict_user_based,
    predict_item_based,
    predict_svd,
    recommend_top_n,
    precision_at_k
)
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- STYLES --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary-color: #7c3aed;
    --primary-dark: #5b21b6;
    --secondary-color: #f97316;
    --accent-color: #fcd34d;
    --success-color: #10b981;
    --background-dark: #121212;
    --background-card: #1e1e1e;
    --background-glass: rgba(255, 255, 255, 0.05);
    --text-primary: #ffffff;
    --text-secondary: #c4c4c4;
    --border-color: rgba(255, 255, 255, 0.1);
    --shadow-glow: 0 0 20px rgba(124, 58, 237, 0.3);
}

/* Body */
body {
    font-family: 'Inter', sans-serif;
    background: var(--background-dark);
    color: var(--text-primary);
}

/* Main Header */
.main-header {
    text-align: center;
    padding: 60px 20px;
}

.main-title {
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}

.main-subtitle {
    font-size: 1.3rem;
    color: var(--text-secondary);
    font-weight: 400;
}

/* Sidebar */
.sidebar .sidebar-content {
    background: var(--background-glass);
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--border-color);
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
    color: white;
    font-weight: 600;
    padding: 14px 28px;
    border-radius: 16px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.5);
}

/* Movies Grid */
.movies-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 24px;
    padding: 40px 20px;
}

.movie-card {
    background: var(--background-card);
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.4s ease;
    border: 1px solid var(--border-color);
}

.movie-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 12px 24px rgba(124, 58, 237, 0.3);
    border-color: var(--primary-color);
}

.movie-poster {
    width: 100%;
    height: 320px;
    object-fit: cover;
    transition: transform 0.4s ease;
}

.movie-card:hover .movie-poster {
    transform: scale(1.08);
}

.movie-info {
    padding: 18px;
}

.movie-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 6px;
}

.movie-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    color: var(--text-secondary);
}

.movie-rating {
    color: var(--accent-color);
    font-weight: 600;
}

.success-message {
    background: linear-gradient(135deg, var(--success-color), #059669);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------- UTILS --------------------
def get_movie_poster(title: str) -> str:
    try:
        clean_title = title.split("(")[0].strip()
        year = title.split("(")[-1].split(")")[0].strip() if "(" in title else None
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": clean_title}
        response = requests.get(url, params=params).json()
        if response.get("results"):
            poster_path = response["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        if year:
            params["year"] = year
            response = requests.get(url, params=params).json()
            if response.get("results"):
                poster_path = response["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return f"https://via.placeholder.com/300x450/1a1a1a/7c3aed?text={clean_title.replace(' ', '+')}"
    except:
        return f"https://via.placeholder.com/300x450/1a1a1a/ff4b4b?text=Error"

def display_movie_card(title: str, rank: int) -> None:
    poster_url = get_movie_poster(title)
    year = "N/A"
    if "(" in title and ")" in title:
        try: year = title.split("(")[-1].split(")")[0]
        except: pass
    rating = round(np.random.uniform(3.5, 4.8), 1)
    st.markdown(f"""
    <div class="movie-card">
        <img src="{poster_url}" class="movie-poster" alt="{title}">
        <div class="movie-info">
            <div class="movie-title">{title}</div>
            <div class="movie-meta">
                <span>{year}</span>
                <div class="movie-rating">{rating}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def run_recommender(data_dir, method, user_id, k, rank, top_n, prec_k, threshold):
    try:
        ratings, movies = load_movielens_100k(data_dir)
        train_df, test_df = make_train_test(ratings, test_size_per_user=5)
        train_mat, uid_to_ix, iid_to_ix, ix_to_uid, ix_to_iid = build_ui_matrix(train_df)

        if user_id not in uid_to_ix:
            st.error(f"User ID {user_id} not found. Try IDs between 1–943.")
            return None, None

        uix = uid_to_ix[user_id]
        if method == "usercf": pred = predict_user_based(train_mat, k=k)
        elif method == "itemcf": pred = predict_item_based(train_mat, k=k)
        elif method == "svd": pred = predict_svd(train_mat, rank=rank)

        rec_indices = recommend_top_n(pred, train_mat, uix, top_n=top_n)
        movie_ids = [ix_to_iid[i] for i in rec_indices]
        recommended_movies = [movies[movies["movie_id"]==mid].iloc[0]["title"] for mid in movie_ids if not movies[movies["movie_id"]==mid].empty]

        precision = precision_at_k(pred, train_mat, test_df, uid_to_ix, iid_to_ix, k=prec_k, threshold=threshold)
        return recommended_movies, precision
    except Exception as e:
        st.error(f"Error running recommender: {str(e)}")
        return None, None

# -------------------- MAIN --------------------
# ...existing code...

def main():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Movie Recommendation System</h1>
        <p class="main-subtitle">Discover your next favorite movie with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Configuration")
        method = st.selectbox(
            "Recommendation Method",
            ["usercf", "itemcf", "svd"],
            format_func=lambda x: {
                "usercf": "User-Based Collaborative Filtering",
                "itemcf": "Item-Based Collaborative Filtering", 
                "svd": "SVD Matrix Factorization"
            }[x]
        )
        user_id = st.number_input("User ID", min_value=1, max_value=943, value=1, step=1)
        k = st.slider("K (neighbors)", 5, 50, 20)
        rank = st.slider("SVD Rank", 5, 100, 20)
        top_n = st.slider("Top-N Recommendations", 5, 20, 10)
        prec_k = st.slider("Precision@K", 5, 20, 10)
        threshold = st.slider("Rating Threshold", 1.0, 5.0, 3.5, 0.5)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Discover Movies", use_container_width=True):
            loading_placeholder = st.empty()
            with loading_placeholder:
                st.markdown("""
                <div style='text-align:center; padding:60px;'>
                    <div style='width:60px;height:60px;border:3px solid #555;border-top:3px solid #7c3aed;border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 20px;'></div>
                    <div style='color:#c4c4c4;font-size:1.1rem;'>Finding your perfect movies...</div>
                </div>
                <style>@keyframes spin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}</style>
                """, unsafe_allow_html=True)
            
            recommended_movies, precision = run_recommender(
                "./ml-100k", method, user_id, k, rank, top_n, prec_k, threshold
            )
            
            loading_placeholder.empty()
            
            if recommended_movies and precision is not None:
                st.markdown(f"""
                <div class="success-message">
                    Precision@{prec_k}: {precision:.4f}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="movies-grid">', unsafe_allow_html=True)
                for movie in recommended_movies:
                    display_movie_card(movie, 0)
                st.markdown('</div>', unsafe_allow_html=True)

# Add this at the bottom of the file to call main()
if __name__ == "__main__":
    main()
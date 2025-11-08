# app.py
# CineMatch — Streamlit Movie Recommender (single-file)
# Requirements: streamlit, pandas, requests, pickle5 (or built-in pickle), concurrent.futures
# Run: pip install streamlit pandas requests && streamlit run app.py

import streamlit as st
import pandas as pd
import requests
import pickle
import re
import random
import concurrent.futures
from typing import Tuple, List

# --- 1. Configuration (from your prompt) ---
API_READ_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzYmQzMGMxZmQ1YTkwNzdkODNlZGU1NDRiNzE5MGEzMCIsIm5iZiI6MTc2MjE1MjU2NS4wODcwMDAxLCJzdWIiOiI2OTA4NTA3NTMxZTQzNThmNDEwODE4MzUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.npzY38JNcrTkUKFDZ41XiZs_CmZsSls3oU63vo8gIIo"
HEADERS = {"accept": "application/json", "Authorization": f"Bearer {API_READ_ACCESS_TOKEN}"}

PICKLE_MOVIES_URL = (
    "https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/"
    "Movie%20Recommender%20System/pickle%20files/movies_dict.pkl"
)
PICKLE_SIM_URL = (
    "https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/"
    "Movie%20Recommender%20System/pickle%20files/similarity.pkl"
)

# --- 2. Utility helpers ---
def format_name(name_string):
    if isinstance(name_string, str):
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', name_string)
    return name_string

# Safe poster fetch (TMDB)
def fetch_poster(movie_id: int) -> str:
    """Uses the provided API token to fetch poster path from TMDB; falls back to placeholder."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
        resp = requests.get(url, headers=HEADERS, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        if data.get("posters"):
            file_path = data["posters"][0].get("file_path")
            if file_path:
                return f"https://image.tmdb.org/t/p/w500{file_path}"
    except Exception:
        pass
    return "https://via.placeholder.com/500x750.png?text=Poster+Not+Available"

def similarity_score(a_tags: List[str], b_tags: List[str]) -> float:
    """Jaccard-like normalized overlap for tag lists (content-based)."""
    a = set([t.lower() for t in (a_tags or [])])
    b = set([t.lower() for t in (b_tags or [])])
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# --- 3. Data loader (cached) ---
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, object, List[str]]:
    """
    Downloads and processes the pickles from GitHub.
    Expects the movies_dict.pkl to be a dict convertible to a DataFrame with columns:
      ['movie_id','title','overview','genres','vote_average','cast','crew', ...]
    similarity.pkl should be a 2D array-like similarity matrix.
    """
    try:
        r1 = requests.get(PICKLE_MOVIES_URL, timeout=15)
        r1.raise_for_status()
        movies_dict = pickle.loads(r1.content)
        movies = pd.DataFrame(movies_dict)

        # Required columns guard
        required_cols = ['movie_id', 'title', 'overview', 'genres', 'vote_average', 'cast', 'crew']
        if not all(c in movies.columns for c in required_cols):
            raise RuntimeError(f"movies_dict.pkl missing required columns: {required_cols}")

        # Normalize director/cast fields
        def extract_director(crew):
            if isinstance(crew, list) and crew:
                return format_name(crew[0])
            return "N/A"

        movies['director'] = movies['crew'].apply(extract_director)
        movies['cast'] = movies['cast'].apply(lambda x: [format_name(n) for n in x] if isinstance(x, list) else [])
        # ensure genres is list
        movies['genres'] = movies['genres'].apply(lambda g: g if isinstance(g, list) else [])

        # genres list
        all_genres = sorted({g for sub in movies['genres'] for g in sub} if not movies['genres'].isnull().all() else [])

        r2 = requests.get(PICKLE_SIM_URL, timeout=15)
        r2.raise_for_status()
        similarity = pickle.loads(r2.content)

        return movies, similarity, all_genres
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
        return pd.DataFrame(), None, []

# --- 4. Recommendation function ---
def recommend(movie_title: str, movies_df: pd.DataFrame, similarity_matrix) -> Tuple[List[str], List[str], List[str], List[float], List[List[str]]]:
    """
    Returns (names, posters, overviews, ratings, genres) for top 8 similar + 2 least-similar (wildcards).
    This function expects `similarity_matrix` to be index-aligned with movies_df.
    """
    if movies_df is None or similarity_matrix is None:
        return [], [], [], [], []
    try:
        idxs = movies_df[movies_df['title'] == movie_title].index
        if len(idxs) == 0:
            return [], [], [], [], []
        idx = idxs[0]
        row = similarity_matrix[idx]
        # row can be list-like of similarity scores
        enumerated = sorted(list(enumerate(row)), key=lambda x: x[1], reverse=True)
        # exclude the same movie
        enumerated = [e for e in enumerated if e[0] != idx]
        top8 = enumerated[:8]
        bottom2 = sorted(enumerated[-2:], key=lambda x: x[1])  # least similar
        picks = top8 + bottom2

        movie_ids, names, overviews, ratings, genres_lists = [], [], [], [], []
        for i, score in picks:
            r = movies_df.iloc[i]
            movie_ids.append(int(r.get('movie_id', -1)))
            names.append(r.get('title', 'Unknown'))
            overviews.append(r.get('overview') if isinstance(r.get('overview'), list) else [str(r.get('overview') or "")])
            ratings.append(float(r.get('vote_average', 0.0)))
            genres_lists.append(r.get('genres', []))

        # parallel fetch posters
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            posters = list(ex.map(fetch_poster, movie_ids))

        return names, posters, overviews, ratings, genres_lists
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return [], [], [], [], []

# --- 5. Small helper UI code for styling (inspired by uploaded demo) ---
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
:root{
  --bg:#0f0f10;
  --card:#181818;
  --border:#333333;
  --primary:#E50914;
  --muted:#9CA3AF;
  --text:#F3F4F6;
}
html,body,.stApp{background:var(--bg); color:var(--text); font-family:Montserrat,system-ui,Arial;}
#MainMenu, footer {visibility: hidden;}
h1 {color:var(--primary); text-align:center;}
.movie-grid {display:grid; grid-template-columns: repeat(auto-fill,minmax(220px,1fr)); gap:18px;}
.movie-card {background:var(--card); border:1px solid var(--border); padding:10px; border-radius:10px; transition: transform .18s ease, box-shadow .18s ease; position:relative; overflow:hidden;}
.movie-card:hover {transform: translateY(-6px) scale(1.01); box-shadow: 0 18px 40px rgba(0,0,0,0.6); border-color:var(--primary);}
.movie-title {font-weight:700; margin-top:8px; color:var(--text);}
.movie-rating {position:absolute; left:10px; top:10px; background:#00000088; padding:6px 9px; border-radius:10px; font-weight:700;}
.tag {background:var(--primary); color:#fff; padding:4px 8px; border-radius:999px; margin-right:6px; font-size:12px;}
.small-muted {color:var(--muted); font-size:13px;}
.app-footer {text-align:center; color:var(--muted); margin-top:24px; border-top:1px solid var(--border); padding-top:12px;}
</style>
"""

# --- 6. App UI ---
def main():
    st.set_page_config(page_title="CineMatch Recommender", page_icon="🎬", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    movies, similarity, all_genres = load_data()

    st.title("CineMatch — Movie Recommender")
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        selected_genres = st.multiselect("Genres", options=all_genres)
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.1)
        st.write("---")
        st.write("Tip: pick a genre and press Surprise Me!")

    # Filter movies DataFrame
    def filter_movies(df):
        out = df.copy()
        if selected_genres:
            out = out[out['genres'].apply(lambda gl: any(g in gl for g in selected_genres))]
        if min_rating > 0:
            out = out[out['vote_average'] >= min_rating]
        return out

    filtered_df = filter_movies(movies)
    titles = sorted(filtered_df['title'].tolist())

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Pick a movie (or type to search):", [""] + titles, index=0)
    with col2:
        if st.button("🎲 Surprise Me!"):
            if titles:
                selected = random.choice(titles)
                st.experimental_set_query_params(movie=selected)
                st.success(f"Surprised! Showing: {selected}")
            else:
                st.warning("No movies available for the current filters.")
                selected = None

    if selected:
        st.markdown(f"### Selected: {selected}")
        details = get_movie_details_safe(movies, selected)
        if details:
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(details['poster_url'], use_column_width=True, caption=f"{details['title']} — {details['rating']:.1f}")
            with cols[1]:
                st.markdown(f"**Genres:** " + ", ".join(details['genres']))
                st.markdown(f"**Director:** {details['director']}")
                st.markdown("**Cast:** " + ", ".join(details['cast'][:6]))
                st.markdown("---")
                st.markdown("**Overview**")
                st.write(" ".join(details['overview']) if isinstance(details['overview'], list) else details['overview'])

        # recommendations
        if st.button("Show Recommendations"):
            with st.spinner("Computing recommendations..."):
                names, posters, overviews, ratings, genres_lists = recommend(selected, movies, similarity)
                if not names:
                    st.error("No recommendations found — try another movie.")
                else:
                    st.markdown("#### Top recommendations")
                    # grid of cards
                    cards_html = '<div class="movie-grid">'
                    for i, name in enumerate(names):
                        # mark wildcard positions 8 & 9 (if present)
                        wildcard_tag = ""
                        if i >= 8:
                            wildcard_tag = '<div class="tag" style="background:#7c3aed">Wildcard</div>'
                        genre_str = " ".join(f'<span class="tag">{g}</span>' for g in (genres_lists[i] or [])[:3])
                        card = f"""
                          <div class="movie-card">
                            <div class="movie-rating">⭐ {ratings[i]:.1f}</div>
                            <img src="{posters[i]}" alt="{name}" style="width:100%; border-radius:7px;">
                            <div class="movie-title">{name}</div>
                            <div class="small-muted">{', '.join(genres_lists[i] or [])}</div>
                            <div style="margin-top:8px">{genre_str} {wildcard_tag}</div>
                          </div>
                        """
                        cards_html += card
                    cards_html += "</div>"
                    st.write(cards_html, unsafe_allow_html=True)

    st.markdown('<div class="app-footer">This app demos a content-based recommender (tag overlap + rating). Replace dataset or TMDB hooks for production.</div>', unsafe_allow_html=True)

# small helper to safely get movie details -> returns dict or None
def get_movie_details_safe(movies_df: pd.DataFrame, title: str):
    try:
        row = movies_df[movies_df['title'] == title].iloc[0]
        poster = fetch_poster(int(row.get('movie_id', -1)))
        return {
            "id": int(row.get('movie_id', -1)),
            "title": row.get('title', 'N/A'),
            "overview": row.get('overview', []),
            "genres": row.get('genres', []),
            "rating": float(row.get('vote_average', 0.0)),
            "poster_url": poster,
            "cast": row.get('cast', []),
            "director": row.get('director', 'N/A')
        }
    except Exception:
        return None

if __name__ == "__main__":
    main()

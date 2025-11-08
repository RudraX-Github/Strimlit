# app.py
# CineMatch — Streamlit Movie Recommender (single-file, corrected)
# Requirements: streamlit, pandas, requests
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

def fetch_poster(movie_id: int) -> str:
    """Uses the provided API token to fetch poster path from TMDB; falls back to placeholder."""
    try:
        if movie_id is None or movie_id == -1:
            raise ValueError("Invalid movie id")
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        # simple fields call is usually sufficient; images endpoint can be heavier
        resp = requests.get(url, headers=HEADERS, timeout=6, params={"append_to_response": "images"})
        resp.raise_for_status()
        data = resp.json()
        # prefer poster_path top-level
        file_path = data.get("poster_path")
        if not file_path:
            images = data.get("images", {})
            posters = images.get("posters", [])
            if posters:
                file_path = posters[0].get("file_path")
        if file_path:
            return f"https://image.tmdb.org/t/p/w500{file_path}"
    except Exception:
        pass
    return "https://via.placeholder.com/500x750.png?text=Poster+Not+Available"

def similarity_score(a_tags: List[str], b_tags: List[str]) -> float:
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
        r1 = requests.get(PICKLE_MOVIES_URL, timeout=20)
        r1.raise_for_status()
        movies_dict = pickle.loads(r1.content)
        movies = pd.DataFrame(movies_dict)

        # Required columns guard
        required_cols = ['movie_id', 'title', 'overview', 'genres', 'vote_average', 'cast', 'crew']
        missing = [c for c in required_cols if c not in movies.columns]
        if missing:
            raise RuntimeError(f"movies_dict.pkl missing required columns: {missing}")

        # Normalize director/cast fields
        def extract_director(crew):
            try:
                if isinstance(crew, list):
                    # try to find a person with 'Director' in job or the first entry
                    for member in crew:
                        if isinstance(member, dict) and 'job' in member and member['job'].lower() == 'director':
                            return format_name(member.get('name', 'N/A'))
                    # fallback to first element if it's a str or dict
                    first = crew[0]
                    if isinstance(first, dict):
                        return format_name(first.get('name', 'N/A'))
                    return format_name(str(first))
            except Exception:
                pass
            return "N/A"

        def normalize_cast_field(cast):
            if isinstance(cast, list):
                normalized = []
                for c in cast:
                    if isinstance(c, dict):
                        normalized.append(format_name(c.get('name', '')))
                    else:
                        normalized.append(format_name(str(c)))
                return normalized
            return []

        movies['director'] = movies['crew'].apply(extract_director)
        movies['cast'] = movies['cast'].apply(normalize_cast_field)
        movies['genres'] = movies['genres'].apply(lambda g: g if isinstance(g, list) else [])

        all_genres = sorted({g for sub in movies['genres'] for g in sub} if len(movies) else [])

        r2 = requests.get(PICKLE_SIM_URL, timeout=20)
        r2.raise_for_status()
        similarity = pickle.loads(r2.content)

        return movies, similarity, all_genres
    except Exception as e:
        # Let the caller show an error with details
        raise RuntimeError(f"Error loading data: {e}") from e

# --- 4. Recommendation function ---
def recommend(movie_title: str, movies_df: pd.DataFrame, similarity_matrix) -> Tuple[List[str], List[str], List[str], List[float], List[List[str]]]:
    if movies_df is None or similarity_matrix is None:
        return [], [], [], [], []
    try:
        idxs = movies_df[movies_df['title'] == movie_title].index
        if len(idxs) == 0:
            return [], [], [], [], []
        idx = int(idxs[0])
        row = similarity_matrix[idx]
        enumerated = sorted(list(enumerate(row)), key=lambda x: x[1], reverse=True)
        enumerated = [e for e in enumerated if e[0] != idx]
        top8 = enumerated[:8]
        bottom2 = sorted(enumerated[-2:], key=lambda x: x[1])
        picks = top8 + bottom2

        movie_ids, names, overviews, ratings, genres_lists = [], [], [], [], []
        for i, score in picks:
            r = movies_df.iloc[int(i)]
            movie_ids.append(int(r.get('movie_id', -1)))
            names.append(r.get('title', 'Unknown'))
            ov = r.get('overview') or ""
            # ensure overview is a string
            if isinstance(ov, list):
                ov = " ".join([str(x) for x in ov])
            overviews.append(str(ov))
            ratings.append(float(r.get('vote_average', 0.0)))
            genres_lists.append(r.get('genres', []))

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            posters = list(ex.map(fetch_poster, movie_ids))

        return names, posters, overviews, ratings, genres_lists
    except Exception:
        return [], [], [], [], []

# --- 5. Styling (closed properly) ---
PAGE_CSS = """<style>
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
</style>"""

# --- 6. App UI ---
def main():
    st.set_page_config(page_title="CineMatch Recommender", page_icon="🎬", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    try:
        movies, similarity, all_genres = load_data()
    except Exception as e:
        st.error(str(e))
        return

    st.title("CineMatch — Movie Recommender")
    with st.sidebar:
        st.header("Filters")
        selected_genres = st.multiselect("Genres", options=all_genres)
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.1)
        st.write("---")
        st.write("Tip: pick a genre and press Surprise Me!")

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
                st.write(details['overview'] if isinstance(details['overview'], str) else " ".join(details['overview']))

        if st.button("Show Recommendations"):
            names, posters, overviews, ratings, genres_lists = recommend(selected, movies, similarity)
            if not names:
                st.error("No recommendations found — try another movie.")
            else:
                st.markdown("#### Top recommendations")
                cards_html = '<div class="movie-grid">'
                for i, name in enumerate(names):
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

def get_movie_details_safe(movies_df: pd.DataFrame, title: str):
    try:
        row = movies_df[movies_df['title'] == title].iloc[0]
        poster = fetch_poster(int(row.get('movie_id', -1)))
        ov = row.get('overview') or ""
        if isinstance(ov, list):
            ov = " ".join([str(x) for x in ov])
        return {
            "id": int(row.get('movie_id', -1)),
            "title": row.get('title', 'N/A'),
            "overview": str(ov),
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

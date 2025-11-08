# app.py
# CineMatch — Streamlit Movie Recommender (single-file, updated: top10 + rating formatting + HTML fixes)
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

# --- Configuration (from your prompt) ---
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

# --- Utility helpers ---
def format_name(name_string):
    if isinstance(name_string, str):
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', name_string)
    return name_string

def fetch_poster(movie_id: int) -> str:
    """Fetch poster path from TMDB; fallback to placeholder."""
    try:
        if movie_id is None or movie_id == -1:
            raise ValueError("Invalid movie id")
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        resp = requests.get(url, headers=HEADERS, timeout=6, params={"append_to_response": "images"})
        resp.raise_for_status()
        data = resp.json()
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

# --- Data loader (cached) ---
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

        required_cols = ['movie_id', 'title', 'overview', 'genres', 'vote_average', 'cast', 'crew']
        missing = [c for c in required_cols if c not in movies.columns]
        if missing:
            raise RuntimeError(f"movies_dict.pkl missing required columns: {missing}")

        def extract_director(crew):
            try:
                if isinstance(crew, list):
                    for member in crew:
                        if isinstance(member, dict) and 'job' in member and member['job'].lower() == 'director':
                            return format_name(member.get('name', 'N/A'))
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

        all_genres = sorted({g for sub in movies['genres'] for g in sub}) if len(movies) else []

        r2 = requests.get(PICKLE_SIM_URL, timeout=20)
        r2.raise_for_status()
        similarity = pickle.loads(r2.content)

        return movies, similarity, all_genres
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}") from e

# --- Recommendation: top 8 similar + 2 opposites (total 10) ---
def recommend_top10(movie_title: str, movies_df: pd.DataFrame, similarity_matrix) -> Tuple[List[str], List[str], List[str], List[float], List[List[str]]]:
    """
    Returns top 10: first 8 most similar (highest score) then 2 most dissimilar (lowest score).
    Robust if there are fewer movies than 10.
    """
    if movies_df is None or similarity_matrix is None:
        return [], [], [], [], []
    try:
        idxs = movies_df[movies_df['title'] == movie_title].index
        if len(idxs) == 0:
            return [], [], [], [], []
        idx = int(idxs[0])
        row = similarity_matrix[idx]
        n_movies = len(row)

        enumerated = list(enumerate(row))
        # exclude the same item
        enumerated = [e for e in enumerated if e[0] != idx]

        # sort descending for most similar
        most_similar = sorted(enumerated, key=lambda x: x[1], reverse=True)
        top_sim = most_similar[:8]  # may be shorter if not enough movies

        # sort ascending to get most dissimilar
        least_similar = sorted(enumerated, key=lambda x: x[1])
        # ensure dissimilar picks are not already in top_sim (by index), pick up to 2
        top_sim_idxs = {i for i, _ in top_sim}
        opposites = []
        for i, s in least_similar:
            if i not in top_sim_idxs:
                opposites.append((i, s))
            if len(opposites) >= 2:
                break

        picks = top_sim + opposites

        # if still fewer than 10 (small dataset), fill with next-best unique items
        picked_idxs = {i for i, _ in picks}
        if len(picks) < 10:
            for i, s in most_similar:
                if i not in picked_idxs:
                    picks.append((i, s))
                    picked_idxs.add(i)
                if len(picks) >= 10:
                    break

        # build return lists
        movie_ids, names, overviews, ratings, genres_lists = [], [], [], [], []
        for i, score in picks:
            r = movies_df.iloc[int(i)]
            movie_ids.append(int(r.get('movie_id', -1)))
            names.append(r.get('title', 'Unknown'))
            ov = r.get('overview') or ""
            if isinstance(ov, list):
                ov = " ".join([str(x) for x in ov])
            overviews.append(str(ov))
            # ensure rating is float and normalized to single decimal
            try:
                ratings.append(float(r.get('vote_average', 0.0)))
            except Exception:
                ratings.append(0.0)
            genres_lists.append(r.get('genres', []) or [])

        # fetch posters in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            posters = list(ex.map(fetch_poster, movie_ids))

        return names, posters, overviews, ratings, genres_lists
    except Exception:
        return [], [], [], [], []

# --- CSS / page styling ---
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
.movie-title {font-weight:700; margin-top:8px; color:var(--text); font-size:16px;}
.movie-rating {position:absolute; left:10px; top:10px; background:#00000088; padding:6px 9px; border-radius:10px; font-weight:700;}
.tag {background:var(--primary); color:#fff; padding:4px 8px; border-radius:999px; margin-right:6px; font-size:12px; display:inline-block; margin-bottom:6px;}
.small-muted {color:var(--muted); font-size:13px; margin-top:6px; display:block;}
.app-footer {text-align:center; color:var(--muted); margin-top:24px; border-top:1px solid var(--border); padding-top:12px;}
</style>"""

# --- App UI ---
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
                st.markdown(f"**Genres:** " + (", ".join(details['genres']) if details['genres'] else "Unknown"))
                st.markdown(f"**Director:** {details['director']}")
                cast_display = ", ".join(details['cast'][:6]) if details['cast'] else "N/A"
                st.markdown("**Cast:** " + cast_display)
                st.markdown("---")
                st.markdown("**Overview**")
                st.write(details['overview'] or "No overview available.")

        if st.button("Show Recommendations"):
            names, posters, overviews, ratings, genres_lists = recommend_top10(selected, movies, similarity)
            if not names:
                st.error("No recommendations found — try another movie.")
            else:
                st.markdown("#### Top 10 recommendations (8 similar + 2 opposites)")
                cards_html = '<div class="movie-grid">'
                for i, name in enumerate(names):
                    # determine if this item is a wildcard (opposite) - we placed opposites at positions 8 and 9
                    is_wildcard = (i >= 8)
                    # build genre badges (limit to 4)
                    genre_tags = ""
                    if genres_lists[i]:
                        for g in (genres_lists[i] or [])[:4]:
                            genre_tags += f'<span class="tag">{g}</span>'
                    else:
                        genre_tags = '<span class="small-muted">No genres</span>'
                    # rating formatted with 1 decimal
                    rating_str = f"{ratings[i]:.1f}"
                    wildcard_html = '<span class="tag" style="background:#7c3aed">Wildcard</span>' if is_wildcard else ""
                    # small-muted summary (comma-separated genres)
                    small_muted = ", ".join(genres_lists[i] or []) or "Unknown"
                    card = f"""
                      <div class="movie-card">
                        <div class="movie-rating">⭐ {rating_str}</div>
                        <img src="{posters[i]}" alt="{name}" style="width:100%; border-radius:7px;">
                        <div class="movie-title">{name}</div>
                        <div class="small-muted">{small_muted}</div>
                        <div style="margin-top:8px">{genre_tags} {wildcard_html}</div>
                      </div>
                    """
                    cards_html += card
                cards_html += "</div>"
                st.write(cards_html, unsafe_allow_html=True)

    st.markdown('<div class="app-footer">This app demos a content-based recommender (8 similar + 2 opposite picks). Replace dataset or TMDB hooks for production.</div>', unsafe_allow_html=True)

def get_movie_details_safe(movies_df: pd.DataFrame, title: str):
    try:
        row = movies_df[movies_df['title'] == title].iloc[0]
        poster = fetch_poster(int(row.get('movie_id', -1)))
        ov = row.get('overview') or ""
        if isinstance(ov, list):
            ov = " ".join([str(x) for x in ov])
        try:
            rating = float(row.get('vote_average', 0.0))
        except Exception:
            rating = 0.0
        return {
            "id": int(row.get('movie_id', -1)),
            "title": row.get('title', 'N/A'),
            "overview": str(ov),
            "genres": row.get('genres', []) or [],
            "rating": rating,
            "poster_url": poster,
            "cast": row.get('cast', []) or [],
            "director": row.get('director', 'N/A')
        }
    except Exception:
        return None

if __name__ == "__main__":
    main()

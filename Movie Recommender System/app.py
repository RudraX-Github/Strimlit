import streamlit as st
import pickle
import pandas as pd
import requests
import re
import random
import concurrent.futures
from typing import List, Tuple, Optional

# Configuration
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

# --- Styling inspired by CineMatch.html ---
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
        :root{
            --primary: #E50914;
            --bg: #0f0f0f;
            --panel: #0f0f0f;
            --muted: #AAAAAA;
            --border: #242424;
        }
        html, body, .stApp { background: var(--bg); color: #fff; font-family: Montserrat, system-ui, sans-serif }
        #MainMenu {visibility: hidden} footer {visibility: hidden}
        h1 { color: var(--primary); text-align:center; font-size:2.4rem; margin-bottom:6px }
        .hero { text-align:center; color:#ddd; margin-bottom:18px }
        .panel { background: linear-gradient(180deg,#0f0f0f,#0b0b0b); border:1px solid var(--border); padding:12px; border-radius:12px }
        .chip { background: linear-gradient(90deg,#2a2a2a,#151515); padding:6px 10px; border-radius:999px; display:inline-block; margin:4px; font-size:0.85rem }
        .movie-grid { display:flex; flex-wrap:wrap; gap:14px; }
        .movie-item { width:150px; border-radius:10px; overflow:hidden; background:#111; border:1px solid var(--border); padding:6px }
        .movie-item img { width:100%; height:225px; object-fit:cover; border-radius:6px }
        .movie-title { font-weight:600; font-size:0.9rem; margin-top:8px; text-align:center }
        .topbar { display:flex; justify-content:space-between; align-items:center }
        .about-btn { background:linear-gradient(90deg,var(--primary),#ff5a5a); color:#fff; padding:8px 12px; border-radius:8px; font-weight:700 }
        .muted { color:var(--muted) }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Data loading ---
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, List[List[float]], List[str]]:
    try:
        r = requests.get(PICKLE_MOVIES_URL, timeout=8)
        r.raise_for_status()
        movies_dict = pickle.loads(r.content)
        movies = pd.DataFrame(movies_dict)

        # minimal required fields check
        required_cols = ['movie_id', 'title', 'overview', 'genres', 'vote_average', 'cast', 'crew']
        for col in required_cols:
            if col not in movies.columns:
                movies[col] = None

        def extract_director(crew_list):
            if isinstance(crew_list, list) and len(crew_list) > 0:
                return crew_list[0]
            return 'N/A'

        movies['director'] = movies['crew'].apply(extract_director)
        movies['cast_display'] = movies['cast'].apply(lambda c: ', '.join(c) if isinstance(c, list) else '')

        all_genres = sorted({g for genre_list in movies['genres'] for g in (genre_list or [])})

        r2 = requests.get(PICKLE_SIM_URL, timeout=8)
        r2.raise_for_status()
        similarity = pickle.loads(r2.content)

        return movies, similarity, all_genres
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
        return pd.DataFrame(), [], []

# --- TMDB poster helper ---
def fetch_poster(movie_id: int) -> str:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    try:
        res = requests.get(url, headers=HEADERS, timeout=5)
        res.raise_for_status()
        data = res.json()
        if data.get('posters'):
            fp = data['posters'][0].get('file_path')
            if fp:
                return f"https://image.tmdb.org/t/p/w500/{fp}"
    except Exception:
        pass
    return "https://via.placeholder.com/300x450.png?text=No+Poster"

# --- Recommendation (kept simple) ---
def recommend(movie_title: str, movies_df: pd.DataFrame, similarity_matrix) -> Tuple[List[str], List[str], List[str], List[float]]:
    if movies_df is None or similarity_matrix is None:
        return [], [], [], []
    try:
        idxs = movies_df[movies_df['title'] == movie_title].index
        if len(idxs) == 0:
            return [], [], [], []
        idx = idxs[0]
        distances = similarity_matrix[idx]
        sorted_idx = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)
        top = [i for i,_ in sorted_idx[1:9]]
        names, posters, ratings = [], [], []
        ids = movies_df.iloc[top]['movie_id'].tolist()
        names = movies_df.iloc[top]['title'].tolist()
        ratings = movies_df.iloc[top]['vote_average'].tolist()
        with concurrent.futures.ThreadPoolExecutor() as exc:
            posters = list(exc.map(fetch_poster, ids))
        return names, posters, movies_df.iloc[top]['overview'].tolist(), ratings
    except Exception:
        return [], [], [], []

# --- UI ---

def main():
    st.set_page_config(page_title='CineMatch', page_icon='🎬', layout='wide')
    inject_css()

    movies, similarity, all_genres = load_data()

    # Top bar with About button (replaces edit icon)
    left, right = st.columns([6,1])
    with left:
        st.markdown('<div class="topbar">
  <div><h1>CineMatch</h1><div class="hero muted">Discover movies by genre & explore recommendations</div></div>
</div>', unsafe_allow_html=True)
    with right:
        if st.button('About', key='about_btn'):
            st.modal('About CineMatch', True)

    # Sidebar: only genre selection and Surprise Me
    with st.sidebar:
        st.header('Filter by Genre')
        selected_genres = st.multiselect('Select genre(s)', options=all_genres, key='genres_multiselect')
        st.markdown('')
        if st.button('🎲 Surprise Me!', key='surprise'):
            # pick a random movie from filtered set if genres chosen, else random overall
            pool = movies
            if selected_genres:
                pool = movies[movies['genres'].apply(lambda gl: any(g in (gl or []) for g in selected_genres))]
            if not pool.empty:
                pick = pool.sample(1).iloc[0]['title']
                st.session_state['movie_selector'] = pick
        st.markdown('---')
        st.markdown('<div class="muted">Tip: select one or more genres to browse all matching movies in the main view.</div>', unsafe_allow_html=True)

    # Main area: behavior changed to genre-driven browsing
    st.markdown('---')
    st.subheader('Browse movies')

    # If genres are selected: show all movies in those genres (grid)
    if st.session_state.get('genres_multiselect'):
        genres = st.session_state['genres_multiselect']
        mask = movies['genres'].apply(lambda gl: any(g in (gl or []) for g in genres))
        browse_df = movies[mask]
        st.markdown(f"### Showing {len(browse_df)} movies for: {' • '.join(genres)}")

        # grid display (simple)
        cols = st.columns(6)
        cnt = 0
        ids = browse_df['movie_id'].tolist()
        titles = browse_df['title'].tolist()
        # prefetch posters for visible items (limit 60)
        preview_ids = ids[:60]
        with concurrent.futures.ThreadPoolExecutor() as exc:
            posters = list(exc.map(fetch_poster, preview_ids))
        poster_map = {mid: p for mid, p in zip(preview_ids, posters)}

        for idx, row in browse_df.reset_index().iterrows():
            col = cols[cnt % 6]
            with col:
                poster = poster_map.get(row['movie_id'], 'https://via.placeholder.com/300x450.png?text=No+Poster')
                st.markdown(f"<div class=\"movie-item\">
  <img src=\"{poster}\"/>
  <div class=\"movie-title\">{row['title']}</div>
</div>", unsafe_allow_html=True)
                if st.button('View', key=f'view_{row.movie_id}'):
                    st.session_state['movie_selector'] = row['title']
            cnt += 1
            if cnt >= 60:
                break

    else:
        st.info('Select a genre from the sidebar to browse movies.')

    # Show details & recommendations for selected movie (if any)
    if st.session_state.get('movie_selector'):
        sel = st.session_state['movie_selector']
        mrow = movies[movies['title'] == sel]
        if not mrow.empty:
            m = mrow.iloc[0]
            st.markdown('---')
            st.subheader(f"{m['title']} — Details & Recommendations")
            c1, c2 = st.columns([1,2])
            with c1:
                st.image(fetch_poster(m['movie_id']), width=220)
                st.markdown(f"**Rating:** {m.get('vote_average', 'N/A')}")
                st.markdown(f"**Director:** {m.get('director', 'N/A')}")
            with c2:
                st.markdown('**Overview**')
                st.write(' '.join(m.get('overview') or []))
                st.markdown('**Cast**')
                st.write(m.get('cast_display', ''))

            # Recommendations
            if st.button('Show Similar Movies', key='rec_btn'):
                names, posters, overviews, ratings = recommend(m['title'], movies, similarity)
                if names:
                    cols = st.columns(4, gap='large')
                    for i in range(min(4, len(names))):
                        with cols[i]:
                            st.image(posters[i], width=160)
                            st.markdown(f"**{names[i]}**")
                            st.markdown(f"⭐ {ratings[i]}")
                            if st.button('Like this', key=f'like_{i}'):
                                st.session_state['movie_selector'] = names[i]
                else:
                    st.info('No similar movies found.')

    # Development Journey — concise
    st.markdown('---')
    st.header('Development Journey')
    st.markdown('''
    - Converted notebook artifacts into a Streamlit app.
    - Reworked UI to be genre-first (per request): selecting genres shows all matching movies in a visual grid.
    - Removed free-text search to focus browsing on genres and discovery.
    - Replaced the edit/toolbar affordance with a dedicated About button in the top bar.
    - Prefetching posters in parallel for smoother browsing.

    If you'd like a different layout (more columns, infinite scroll, or a sort-by-year control), tell me which part to change and I'll update the app.
    ''')

if __name__ == '__main__':
    main()

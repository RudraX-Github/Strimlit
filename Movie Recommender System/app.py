import streamlit as st
import pickle
import pandas as pd
import requests
import re
import random
import concurrent.futures
from typing import List, Tuple, Optional

# =====================================
# CineMatch — Streamlit App (Refactored)
# =====================================
# Single-file Streamlit app inspired by the provided `app.py` and
# the visual / UX references from `CineMatch.html`.
#
# Features included:
# - Modular structure (data loading, recommendation engine, helpers, UI)
# - Enhanced styling using custom CSS (dark theme + CineMatch feel)
# - Genre filter, "Surprise Me" button, recommendation tabs
# - Rich movie details (poster, cast, director, rating, overview)
# - Development Journey section documenting design decisions, integration
#
# NOTE: This file re-uses the TMDB read token and GitHub raw pickle URLs
# embedded in the original prototype. Replace tokens / URLs in production.
# =====================================

# --- 1. Configuration ---
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

# --- 2. Styling (inspired by CineMatch.html) ---
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
        :root{
            --primary: #E50914;
            --primary-hover: #F61A25;
            --bg: #0f0f0f;
            --panel: #171717;
            --muted: #AAAAAA;
            --border: #2f2f2f;
            --glass: rgba(255,255,255,0.02);
        }
        html, body, .stApp { background: var(--bg); color: #fff; font-family: Montserrat, system-ui, sans-serif }
        #MainMenu {visibility: hidden} footer {visibility: hidden}
        .stButton>button { border-radius: 10px }
        h1 { color: var(--primary); text-align:center; font-size:2.6rem }
        h2 { color: #fff; margin-top: 0 }
        .card { background: linear-gradient(180deg, #161616, #111111); border: 1px solid var(--border); padding: 16px; border-radius: 12px }
        .movie-poster { border-radius: 8px; box-shadow: 0 8px 24px rgba(0,0,0,0.6); }
        .tag { background: var(--primary); padding:6px 12px; border-radius:14px; font-weight:600; font-size:0.8rem }
        .detail-list { color: #ddd }
        .movie-card { background: linear-gradient(145deg,#151515,#1f1f1f); border:1px solid var(--border); border-radius:10px; padding:8px; }
        .movie-card img { width:100%; border-radius:6px; }
        .movie-rating { position:absolute; top:12px; left:12px; background:#0008; padding:6px 8px; border-radius:8px; font-weight:700 }
        .app-footer { color: var(--muted); text-align:center; font-size:0.85rem; margin-top:20px }
        .timeline { border-left:4px solid var(--border); margin-left:10px; padding-left:18px }
        .timeline h4 { margin-bottom:4px }
        .muted { color: var(--muted) }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- 3. Data loading & Processing ---
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, List[List[float]], List[str]]:
    """Load movies DataFrame and similarity matrix from remote pickles.

    Returns (movies_df, similarity_matrix, all_genres)
    """
    def format_name(name_string):
        if isinstance(name_string, str):
            return re.sub(r'([a-z])([A-Z])', r'\1 \2', name_string)
        return name_string

    try:
        r = requests.get(PICKLE_MOVIES_URL, timeout=8)
        r.raise_for_status()
        movies_dict = pickle.loads(r.content)
        movies = pd.DataFrame(movies_dict)

        required_cols = ['movie_id', 'title', 'overview', 'genres', 'vote_average', 'cast', 'crew']
        if not all(col in movies.columns for col in required_cols):
            raise ValueError(f"Missing expected columns in movies pickle: {required_cols}")

        # director extraction (first crew member assumed to be director in this dataset)
        def extract_director_from_list(crew_list):
            if isinstance(crew_list, list) and len(crew_list) > 0:
                return format_name(crew_list[0])
            return 'N/A'

        movies['director'] = movies['crew'].apply(extract_director_from_list)
        movies['cast'] = movies['cast'].apply(lambda names: [format_name(n) for n in names] if isinstance(names, list) else [])

        all_genres = sorted({g for genre_list in movies['genres'] for g in (genre_list or [])})

        r2 = requests.get(PICKLE_SIM_URL, timeout=8)
        r2.raise_for_status()
        similarity = pickle.loads(r2.content)

        return movies, similarity, all_genres

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
        return pd.DataFrame(), [], []


# --- 4. TMDB Poster Helper ---

def fetch_poster(movie_id: int) -> str:
    """Fetch poster path using TMDB images endpoint; fallback to placeholder."""
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
    return "https://via.placeholder.com/500x750.png?text=Poster+Not+Available"


# --- 5. Recommendation Engine ---

def recommend(movie_title: str, movies_df: pd.DataFrame, similarity_matrix) -> Tuple[List[str], List[str], List[List[str]], List[float], List[List[str]]]:
    """Return 10 recommendations (8 most similar + 2 wildcard least-similar).

    Returns: names, posters, overviews, ratings, genres_lists
    """
    if movies_df is None or similarity_matrix is None:
        return [], [], [], [], []

    try:
        indices = movies_df[movies_df['title'] == movie_title].index
        if len(indices) == 0:
            return [], [], [], [], []
        idx = indices[0]

        distances = similarity_matrix[idx]
        sorted_idx = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)
        # exclude the movie itself (first element if identical)
        top8 = sorted_idx[1:9]
        bottom2 = sorted_idx[-2:]
        selection = top8 + bottom2

        ids, names, overviews, ratings, genres_lists = [], [], [], [], []
        for i, _ in selection:
            row = movies_df.iloc[i]
            ids.append(row.movie_id)
            names.append(row.title)
            overviews.append(row.get('overview', []))
            ratings.append(row.get('vote_average', 0.0))
            genres_lists.append(row.get('genres', []))

        with concurrent.futures.ThreadPoolExecutor() as exc:
            posters = list(exc.map(fetch_poster, ids))

        return names, posters, overviews, ratings, genres_lists

    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return [], [], [], [], []


# --- 6. Helpers for UI state ---

def get_movie_details(movie_df: pd.DataFrame, title: str) -> Optional[dict]:
    try:
        m = movie_df[movie_df['title'] == title].iloc[0]
        return {
            'id': m.movie_id,
            'title': m.title,
            'overview': m.get('overview', []),
            'genres': m.get('genres', []),
            'rating': m.get('vote_average', 0.0),
            'poster_url': fetch_poster(m.movie_id),
            'cast': m.get('cast', []),
            'director': m.get('director', 'N/A')
        }
    except Exception:
        return None


def filter_movies(movies_df: pd.DataFrame, selected_genres: List[str]) -> pd.DataFrame:
    if not selected_genres:
        return movies_df
    def has_all(glist):
        return all(g in (glist or []) for g in selected_genres)
    return movies_df[movies_df['genres'].apply(has_all)]


# --- 7. UI — Main App ---

def main():
    st.set_page_config(page_title='CineMatch Recommender', page_icon='🎬', layout='wide')
    inject_css()

    movies, similarity, all_genres = load_data()

    # Header / Hero
    st.title('CineMatch Movie Recommender')
    st.markdown('''
    **Find movies like the ones you love.** Use genre filters, surprise yourself, or inspect detailed cards with cast, director and overview.
    ''')

    # Sidebar filters (collapsible)
    with st.sidebar:
        st.header('🔎 Filters')
        genre_filter = st.multiselect('Genre', options=all_genres, key='genre_filter')
        min_rating = st.slider('Minimum rating', 0.0, 10.0, 0.0, 0.5, key='min_rating')
        q = st.text_input('Search title or cast', '', key='search_q')
        st.markdown('---')
        if st.button('Reset filters'):
            st.session_state.genre_filter = []
            st.session_state.min_rating = 0.0
            st.session_state.search_q = ''

    # Apply filters to main movie list
    filtered_df = filter_movies(movies, st.session_state.get('genre_filter', []))
    if st.session_state.get('min_rating', 0.0) > 0:
        filtered_df = filtered_df[filtered_df['vote_average'] >= st.session_state['min_rating']]
    qv = (st.session_state.get('search_q') or '').strip().lower()
    if qv:
        mask = filtered_df['title'].str.lower().str.contains(qv) | filtered_df['cast'].apply(lambda cl: ' '.join(cl).lower().find(qv) >= 0) | filtered_df['overview'].apply(lambda o: ' '.join(o).lower().find(qv) >= 0)
        filtered_df = filtered_df[mask]

    # Main selection
    st.markdown('---')
    st.header('Choose a movie you like')
    col1, col2 = st.columns([3, 1])
    titles_list = sorted(filtered_df['title'].tolist())
    selected = None
    if titles_list:
        selected = col1.selectbox('Type or pick a title', titles_list, key='movie_selector')
    else:
        st.info('No movies match your filters. Reset or change filters to continue.')

    # Surprise me button
    def surprise_action():
        if titles_list:
            st.session_state.movie_selector = random.choice(titles_list)
    col2.button('🎲 Surprise Me!', on_click=surprise_action)

    # Details and recommendations
    if st.session_state.get('movie_selector'):
        sel = st.session_state.movie_selector
        details = get_movie_details(movies, sel)
        if details:
            st.markdown('---')
            st.subheader(f"You selected: {details['title']}")
            dcol1, dcol2 = st.columns([1, 3])
            with dcol1:
                st.image(details['poster_url'], width=220)
            with dcol2:
                st.markdown(' '.join([f"<span class=\"tag\">{g}</span>" for g in details['genres']]), unsafe_allow_html=True)
                st.markdown(f"**⭐ {details['rating']:.1f} / 10**")
                st.markdown('#### Overview')
                st.write(' '.join(details['overview']))
                st.markdown('#### Cast')
                st.write(', '.join(details['cast']))
                st.markdown('#### Director')
                st.write(details['director'])

            # Recommendation action
            if st.button('Show Recommendations', type='primary'):
                with st.spinner('Computing recommendations...'):
                    names, posters, overviews, ratings, genres_lists = recommend(sel, movies, similarity)

                if names:
                    tab1, tab2 = st.tabs(['Top 5', 'More to Explore'])
                    with tab1:
                        cols = st.columns(5, gap='large')
                        for i in range(5):
                            with cols[i]:
                                st.markdown(f"<div class=\"movie-card\">\n  <div class=\"movie-rating\">⭐ {ratings[i]:.1f}</div>\n  <img src=\"{posters[i]}\" alt=\"{names[i]}\" />\n  <div style=\"padding-top:8px\">\n    <div style=\"font-weight:600\">{names[i]}</div>\n  </div>\n</div>", unsafe_allow_html=True)
                                with st.expander('Details'):
                                    st.markdown(' '.join([f"<span class=\\\"tag\\\">{g}</span>" for g in genres_lists[i]]), unsafe_allow_html=True)
                                    st.write(' '.join(overviews[i]))
                                if st.button('Find movies like this', key=f'like_{i}'):
                                    st.session_state.movie_selector = names[i]

                    with tab2:
                        st.info('Includes wildcard picks to encourage discovery.')
                        cols2 = st.columns(5, gap='large')
                        for j in range(5, 10):
                            with cols2[j - 5]:
                                st.markdown(f"<div class=\"movie-card\">\n  <div class=\"movie-rating\">⭐ {ratings[j]:.1f}</div>\n  <img src=\"{posters[j]}\" alt=\"{names[j]}\" />\n  <div style=\"padding-top:8px\">\n    <div style=\"font-weight:600\">{names[j]}</div>\n  </div>\n</div>", unsafe_allow_html=True)
                                with st.expander('Details'):
                                    st.markdown(' '.join([f"<span class=\\\"tag\\\">{g}</span>" for g in genres_lists[j]]), unsafe_allow_html=True)
                                    st.write(' '.join(overviews[j]))
                                if st.button('Find movies like this', key=f'like2_{j}'):
                                    st.session_state.movie_selector = names[j]

                else:
                    st.error('No recommendations found for this movie.')

    else:
        st.info('Pick a movie from the dropdown to view details and recommendations.')

    # Footer / About
    st.markdown('---')
    st.markdown('<div class=\"app-footer\">This app uses a content-based similarity model (overview, genres, cast, crew) to suggest movies. Posters are fetched from TMDB where available.</div>', unsafe_allow_html=True)

    # Development Journey — Documentation section
    st.markdown('\n---\n')
    st.header('The Development Journey')
    st.markdown('''
    This section documents the integration and design decisions made while converting a Jupyter notebook prototype into this Streamlit app.

    **Phases:**

    - **Phase 1 — Notebook Prototype:** The recommendation logic and dataset shaping (tags, overviews, cast/crew extraction) were prototyped in a Jupyter notebook. That produced `movies_dict.pkl` and `similarity.pkl` files.

    - **Phase 2 — Port & Refactor:** The notebook code was ported into a single-file Streamlit app. Functions were modularized (data loading, poster fetching, recommendation logic) to make testing and caching straightforward.

    - **Phase 3 — UX & Visual Polish:** Styling and layout were upgraded to match the CineMatch design language: dark mode, glass panels, tag chips, 3D-feel movie cards, and a prominent hero. Effort was made to keep the UI responsive and usable in `wide` layout.

    - **Phase 4 — Performance & Resilience:** Network operations (pickles and poster fetches) use timeouts and error-handling. Poster fetching is done in parallel to improve perceived speed. Data is cached to minimize repeated downloads.

    - **Phase 5 — Usability Additions:** Added `Surprise Me!` random picks, a collapsible set of genre filters, search-by-title/cast, and wildcard picks (least similar) to encourage discovery.

    **Why these decisions?**
    - Modular code makes it easier to replace components (e.g., swap in a database or a newer model).
    - Caching removes friction for repeat users.
    - Visual cues (tags, rating badges) quickly communicate why a recommendation was surfaced.

    **How the integration happened:**
    1. Extracted dataset loading cells from the notebook and wrapped them into `load_data()` with caching.
    2. Pulled the recommendation ranking logic and wrapped into `recommend()` which now returns poster URLs via parallel fetches.
    3. Rewrote the original CSS + component ideas from `CineMatch.html` into Streamlit-safe CSS and component layouts (columns / tabs / expanders).

    **Future improvements:**
    - Replace pickle-based dataset with a proper backend (SQL or object store) for large datasets.
    - Add user-level preferences saved to a DB or local storage for personalized sessions.
    - Add hybrid models (collaborative + content-based) and AB testing to improve recommendation quality.

    ''')


if __name__ == '__main__':
    main()

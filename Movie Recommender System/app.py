import streamlit as st
import pickle
import pandas as pd
import requests
import random
import concurrent.futures
from typing import List, Tuple

# --- Configuration ---
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

# --- Styling (cinematic wow + loader ball) ---
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        :root{--accent:#E50914;--accent2:#ff6b6b;--bg1:#050505;--bg2:#0b0b0b}
        html, body, .stApp{background: radial-gradient(1200px 600px at 10% 10%, rgba(229,9,20,0.06), transparent), linear-gradient(180deg,var(--bg1),var(--bg2)); color:#fff; font-family:Inter,system-ui,sans-serif}
        .hero {text-align:center; padding-top:8px}
        .cine-title{font-size:3rem; color:var(--accent); font-weight:800; letter-spacing:1px; margin-bottom:6px;}
        .cine-sub{color:#cfcfcf; margin-bottom:18px}
        /* neon underline */
        .cine-title:after{content:''; display:block; height:6px; width:180px; margin:12px auto 0; border-radius:999px; background:linear-gradient(90deg,var(--accent),var(--accent2)); filter:blur(10px); opacity:0.9}
        /* cards */
        .movie-card{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.03); padding:8px; border-radius:12px; transition:transform .22s, box-shadow .22s}
        .movie-card:hover{transform:translateY(-8px); box-shadow:0 20px 40px rgba(0,0,0,0.6)}
        .poster{width:100%; height:260px; object-fit:cover; border-radius:8px}
        .movie-title{font-weight:700; margin-top:8px; font-size:0.95rem}
        .rating{background:var(--accent); padding:6px 8px; border-radius:8px; color:#fff; font-weight:700; display:inline-block; margin-top:6px}
        /* compact recommendations button */
        .rec-btn{display:inline-block; padding:6px 8px; background:linear-gradient(90deg,var(--accent),var(--accent2)); color:#fff; border-radius:999px; font-weight:700; cursor:pointer; border:none}
        /* loader ball */
        .loader-wrap{display:flex;justify-content:center;padding:18px}
        .loader-ball{width:22px;height:22px;border-radius:50%;background:linear-gradient(90deg,var(--accent),var(--accent2));animation:ballBounce 0.9s infinite}
        @keyframes ballBounce{0%{transform:translateY(0)}50%{transform:translateY(-18px)}100%{transform:translateY(0)}}
        /* responsive tweaks */
        @media (max-width:768px){ .poster{height:200px} .cine-title{font-size:2rem} }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Data loading ---
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, List[List[float]], List[str]]:
    import pickle, requests
    r = requests.get(PICKLE_MOVIES_URL, timeout=10)
    r.raise_for_status()
    movies_dict = pickle.loads(r.content)
    movies = pd.DataFrame(movies_dict)

    r2 = requests.get(PICKLE_SIM_URL, timeout=10)
    r2.raise_for_status()
    similarity = pickle.loads(r2.content)

    movies['overview'] = movies['overview'].apply(lambda o: o or [])
    movies['cast'] = movies['cast'].apply(lambda c: c or [])
    movies['cast_display'] = movies['cast'].apply(lambda c: ', '.join(c) if isinstance(c, list) else str(c))
    movies['genres'] = movies['genres'].apply(lambda g: g or [])

    all_genres = sorted({g for genre_list in movies['genres'] for g in (genre_list or [])})
    return movies, similarity, all_genres

# --- Poster fetch ---

def fetch_poster(movie_id: int) -> str:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    try:
        res = requests.get(url, headers=HEADERS, timeout=6)
        res.raise_for_status()
        data = res.json()
        if data.get('posters'):
            fp = data['posters'][0].get('file_path')
            if fp:
                return f"https://image.tmdb.org/t/p/w500{fp}"
    except Exception:
        pass
    return "https://via.placeholder.com/300x450.png?text=No+Poster"

# --- Recommend ---

def recommend(title: str, movies: pd.DataFrame, similarity) -> Tuple[List[str], List[str], List[float], List[str]]:
    matches = movies[movies['title'] == title]
    if matches.empty:
        return [], [], [], []
    idx = matches.index[0]
    distances = similarity[idx]
    indices = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:9]
    ids = [movies.iloc[i[0]].movie_id for i in indices]
    names = [movies.iloc[i[0]].title for i in indices]
    ratings = [movies.iloc[i[0]].vote_average for i in indices]
    overviews = [movies.iloc[i[0]].overview for i in indices]
    with concurrent.futures.ThreadPoolExecutor() as ex:
        posters = list(ex.map(fetch_poster, ids))
    return names, posters, ratings, overviews

# --- Helpers ---

def set_selected_movie_from_query():
    params = st.experimental_get_query_params()
    if 'selected_movie' in params:
        val = params['selected_movie'][0]
        st.session_state['selected_movie'] = val

# --- Main App ---

def main():
    st.set_page_config(page_title='CineMatch', page_icon='🎬', layout='wide')
    inject_css()

    movies, similarity, all_genres = load_data()
    set_selected_movie_from_query()

    # Header
    st.markdown("""
    <div class='hero'>
      <div class='cine-title'>🎬 CineMatch</div>
      <div class='cine-sub'>Find your next favorite movie — search by title, cast or genre</div>
    </div>
    """, unsafe_allow_html=True)

    # Single unified output area: determine results based on query + genres
    search_col, genre_col = st.columns([2,1])

    with search_col:
        query = st.text_input('Search by title / cast / keyword', value=st.session_state.get('last_query',''))
        if query:
            st.session_state['last_query'] = query
    with genre_col:
        selected_genres = st.multiselect('Filter genres (optional)', options=all_genres, default=st.session_state.get('genres',[]), key='genres')

    # Determine combined results
    def compute_results():
        df = movies
        q = (st.session_state.get('last_query') or '').strip().lower()
        genres = st.session_state.get('genres') or []
        if q:
            mask_q = df['title'].str.lower().str.contains(q) | df['cast_display'].str.lower().str.contains(q) | df['overview'].apply(lambda o: ' '.join(o).lower()).str.contains(q)
            df = df[mask_q]
        if genres:
            mask_g = df['genres'].apply(lambda gl: any(g in (gl or []) for g in genres))
            df = df[mask_g]
        return df

    results = compute_results()

    st.markdown(f"**Showing {len(results)} results**")

    # Display grid of results (single output area)
    cols = st.columns(4)
    poster_ids = results['movie_id'].tolist()
    preview_ids = poster_ids[:48]
    with concurrent.futures.ThreadPoolExecutor() as ex:
        posters = list(ex.map(fetch_poster, preview_ids))
    poster_map = {mid: p for mid, p in zip(preview_ids, posters)}

    for i, (_, row) in enumerate(results.head(48).iterrows()):
        col = cols[i % 4]
        with col:
            poster = poster_map.get(row['movie_id'], 'https://via.placeholder.com/300x450.png?text=No+Poster')
            st.markdown(f"<div class='movie-card'><img class='poster' src='{poster}' alt='poster' /><div class='movie-title'>{row.title}</div><div class='rating'>⭐ {row.vote_average}</div></div>", unsafe_allow_html=True)
            # compact recommendation trigger and details trigger
            c1, c2 = st.columns([3,1])
            with c1:
                if st.button('Details', key=f'detail_{row.movie_id}'):
                    st.session_state['selected_movie'] = row.title
            with c2:
                # small circular rec button
                if st.button('✨', key=f'rec_{row.movie_id}', help='Show recommendations for this movie'):
                    # show loader ball while computing
                    loader = st.empty()
                    loader.markdown("<div class='loader-wrap'><div class='loader-ball'></div></div>", unsafe_allow_html=True)
                    names, posters_rec, ratings_rec, overviews_rec = recommend(row.title, movies, similarity)
                    loader.empty()
                    st.session_state['cur_recs'] = {'names':names,'posters':posters_rec,'ratings':ratings_rec,'overviews':overviews_rec}

    # Selected movie details area (right below grid)
    if st.session_state.get('selected_movie'):
        sel = st.session_state['selected_movie']
        mdf = movies[movies['title'] == sel]
        if not mdf.empty:
            m = mdf.iloc[0]
            st.markdown('---')
            st.markdown(f"## 🎥 {m.title}  <span style='float:right' class='rating'>⭐ {m.vote_average}</span>", unsafe_allow_html=True)
            c1, c2 = st.columns([1,2])
            with c1:
                st.image(fetch_poster(m.movie_id), width=260)
            with c2:
                st.markdown('**Overview**')
                st.write(' '.join(m.overview))
                st.markdown('**Cast**')
                st.write(m.cast_display)
                if st.button('Show Recommendations for this movie', key='detail_rec_btn'):
                    loader = st.empty()
                    loader.markdown("<div class='loader-wrap'><div class='loader-ball'></div></div>", unsafe_allow_html=True)
                    names, posters_rec, ratings_rec, overviews_rec = recommend(m.title, movies, similarity)
                    loader.empty()
                    st.session_state['cur_recs'] = {'names':names,'posters':posters_rec,'ratings':ratings_rec,'overviews':overviews_rec}

    # Render recommendations panel
    if st.session_state.get('cur_recs'):
        recs = st.session_state['cur_recs']
        st.markdown('---')
        st.subheader('✨ Recommendations')
        cols = st.columns(4)
        for i in range(len(recs['names'])):
            with cols[i % 4]:
                st.image(recs['posters'][i], width=160)
                st.markdown(f"**{recs['names'][i]}**")
                st.markdown(f"<div class='muted'>⭐ {recs['ratings'][i]}</div>", unsafe_allow_html=True)
                if st.button('View', key=f'viewrec_{i}'):
                    st.session_state['selected_movie'] = recs['names'][i]

    # Footer notes
    st.markdown('---')
    st.markdown('**Notes:** single unified output supports searching by name, genre, or both. Compact recommendation buttons (✨) and a loader-ball animation give feedback while recommendations compute.')

if __name__ == '__main__':
    main()
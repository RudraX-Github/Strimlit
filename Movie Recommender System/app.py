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

# --- Styling (cinematic wow + loader ball + entrance animation + floating popup) ---
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
        .cine-title:after{content:''; display:block; height:6px; width:180px; margin:12px auto 0; border-radius:999px; background:linear-gradient(90deg,var(--accent),var(--accent2)); filter:blur(10px); opacity:0.9}
        .movie-card{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.03); padding:8px; border-radius:12px; transition:transform .22s, box-shadow .22s, opacity .4s}
        .movie-card:hover{transform:translateY(-8px); box-shadow:0 20px 40px rgba(0,0,0,0.6)}
        .poster{width:100%; height:260px; object-fit:cover; border-radius:8px}
        .movie-title{font-weight:700; margin-top:8px; font-size:0.95rem}
        .rating{background:var(--accent); padding:6px 8px; border-radius:8px; color:#fff; font-weight:700; display:inline-block; margin-top:6px}
        .rec-overlay{position:relative}
        .rec-circle{position:absolute; right:12px; top:12px; width:44px; height:44px; border-radius:50%; background:linear-gradient(90deg,var(--accent),var(--accent2)); display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; box-shadow:0 8px 20px rgba(229,9,20,0.18); cursor:pointer}
        .loader-wrap{display:flex;justify-content:center;padding:18px}
        .loader-ball{width:22px;height:22px;border-radius:50%;background:linear-gradient(90deg,var(--accent),var(--accent2));animation:ballBounce 0.9s infinite}
        @keyframes ballBounce{0%{transform:translateY(0)}50%{transform:translateY(-18px)}100%{transform:translateY(0)}}
        @keyframes fadeUp{from{opacity:0; transform:translateY(12px)} to{opacity:1; transform:none}}
        .rec-card{animation:fadeUp .32s ease both}
        .muted{color:#c0c0c0}
        /* floating popup styles */
        .floating-popup{position:fixed; right:24px; top:90px; width:380px; z-index:9999}
        .floating-inner{background:linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.02)); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.04); box-shadow:0 20px 40px rgba(0,0,0,0.6)}
        .popup-close{background:transparent;border:1px solid rgba(255,255,255,0.06); color:#fff; padding:6px 8px; border-radius:8px}
        @media (max-width:768px){ .poster{height:200px} .cine-title{font-size:2rem} .floating-popup{right:8px; width:320px} }
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

# --- Handlers ---
def handle_set_selected(title: str):
    st.session_state['selected_movie'] = title

def handle_recommend_click(title: str, movies, similarity):
    # set a flag — actual computation happens in the main loop so the loader can be shown
    st.session_state['compute_recs_for'] = title

# --- Helpers for query params ---
def set_selected_movie_from_query():
    params = st.experimental_get_query_params()
    if 'selected_movie' in params:
        val = params['selected_movie'][0]
        st.session_state['selected_movie'] = val

def set_popup_from_query():
    params = st.experimental_get_query_params()
    if 'popup_movie' in params:
        val = params['popup_movie'][0]
        st.session_state['popup_movie'] = val

# --- Main App ---
def main():
    st.set_page_config(page_title='CineMatch', page_icon='🎬', layout='wide')
    inject_css()

    movies, similarity, all_genres = load_data()
    set_selected_movie_from_query()
    set_popup_from_query()

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
        if query is not None:
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
            # poster click opens popup via query param
            link = f"?popup_movie={requests.utils.requote_uri(row.title)}"
            st.markdown(f"<a class='poster-link' href='{link}'><div class='movie-card'><img class='poster' src='{poster}' alt='poster' /><div class='movie-title'>{row.title}</div></div></a>", unsafe_allow_html=True)
            # compact row: rating on left, small rec button on right
            rcol1, rcol2 = st.columns([3,1])
            with rcol1:
                st.markdown(f"<div class='rating'>⭐ {row.vote_average}</div>", unsafe_allow_html=True)
            with rcol2:
                # small circular rec button (no text) — uses on_click handler
                st.button('✨', key=f'rec_{row.movie_id}', help='Show recommendations', on_click=handle_recommend_click, args=(row.title, movies, similarity))
            # keep a Details button for accessibility (opens details section)
            st.button('Details', key=f'detail_{row.movie_id}', on_click=handle_set_selected, args=(row.title,))

    # If a recommendation computation was requested, compute it here so the loader can be shown
    if st.session_state.get('compute_recs_for'):
        title_to_compute = st.session_state.get('compute_recs_for')
        loader = st.empty()
        loader.markdown("<div class='loader-wrap'><div class='loader-ball'></div></div>", unsafe_allow_html=True)
        names, posters_rec, ratings_rec, overviews_rec = recommend(title_to_compute, movies, similarity)
        st.session_state['cur_recs'] = {'names':names,'posters':posters_rec,'ratings':ratings_rec,'overviews':overviews_rec}
        # clear the flag
        del st.session_state['compute_recs_for']
        loader.empty()

    # Popup floating card (if poster clicked) — rendered as a fixed-position overlay
    if st.session_state.get('popup_movie'):
        pm = st.session_state['popup_movie']
        pm_row = movies[movies['title'] == pm]
        if not pm_row.empty:
            m = pm_row.iloc[0]
            st.markdown(f"<div class='floating-popup'><div class='floating-inner'><div style='display:flex; gap:12px'><img src='{fetch_poster(m.movie_id)}' style='width:110px; height:160px; object-fit:cover; border-radius:8px' /><div style='flex:1'><div style='font-weight:800; font-size:1rem'>{m.title}</div><div style='color:#cfcfcf; margin-top:6px'>{' '.join(m.overview)[:260]}...</div><div style='margin-top:8px'><span style=\"background:linear-gradient(90deg,var(--accent),var(--accent2)); padding:6px 8px; border-radius:8px; color:#fff; font-weight:700\">⭐ {m.vote_average}</span> <span style=\"margin-left:8px\">
</span></div></div></div></div></div>", unsafe_allow_html=True)
            # functional controls below the floating card
            c1, c2 = st.columns([2,1])
            with c1:
                if st.button('Close popup', key='close_popup'):
                    st.experimental_set_query_params()
                    if 'popup_movie' in st.session_state:
                        del st.session_state['popup_movie']
            with c2:
                if st.button('Recommendations from popup', key='popup_rec', on_click=handle_recommend_click, args=(m.title, movies, similarity)):
                    if 'popup_movie' in st.session_state:
                        del st.session_state['popup_movie']

    # Selected movie details area
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
                st.button('Show Recommendations for this movie', key='detail_rec_btn', on_click=handle_recommend_click, args=(m.title, movies, similarity))

    # Render recommendations panel with entrance animation
    if st.session_state.get('cur_recs'):
        recs = st.session_state['cur_recs']
        st.markdown('---')
        st.subheader('✨ Recommendations')
        cols = st.columns(4)
        for i in range(len(recs['names'])):
            with cols[i % 4]:
                st.markdown(f"<div class='rec-card'><img src='{recs['posters'][i]}' width='160' style='border-radius:8px' /><div style='font-weight:700;margin-top:8px'>{recs['names'][i]}</div><div class='muted'>⭐ {recs['ratings'][i]}</div></div>", unsafe_allow_html=True)
                st.button('View', key=f'viewrec_{i}', on_click=handle_set_selected, args=(recs['names'][i],))

    # Footer notes
    st.markdown('---')
    st.markdown('**Notes:** single unified output supports searching by name, genre, or both. Compact recommendation overlay and animated effects added.')

if __name__ == '__main__':
    main()

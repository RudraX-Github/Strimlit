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

# --- Styling inspired by CineMatch.html (wow factor: animated gradient, glass cards) ---
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        :root{--accent:#E50914;--bg:#070707;--card:#0f0f0f;--muted:#9b9b9b}
        html, body, .stApp{background:linear-gradient(180deg,#070707 0%, #0b0b0b 40%, #080808 100%); color:#fff; font-family:'Poppins',sans-serif}
        /* animated header underline */
        .hero {text-align:center; margin-bottom:12px}
        .cine-title {font-size:2.6rem; font-weight:700; letter-spacing:1px; color:var(--accent)}
        .cine-sub {color:var(--muted); margin-top:6px}
        @keyframes pulseAccent { 0%{filter:drop-shadow(0 0 0 rgba(229,9,20,0.0))} 50%{filter:drop-shadow(0 0 14px rgba(229,9,20,0.18))} 100%{filter:drop-shadow(0 0 0 rgba(229,9,20,0.0))} }
        .cine-title { animation: pulseAccent 3s infinite }
        /* glass card */
        .movie-card { background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(255,255,255,0.03); padding:8px; border-radius:12px; transition: transform .22s ease, box-shadow .22s ease }
        .movie-card:hover{ transform:translateY(-6px); box-shadow:0 12px 30px rgba(0,0,0,0.6) }
        .poster { width:100%; height:250px; object-fit:cover; border-radius:8px }
        .movie-title{ font-weight:600; font-size:0.95rem; margin-top:8px }
        .rating { background:var(--accent); padding:6px 8px; border-radius:8px; display:inline-block; font-weight:700; }
        .chip { display:inline-block; background:rgba(255,255,255,0.03); padding:6px 10px; border-radius:999px; margin:4px; font-size:0.85rem }
        .topbar { display:flex; justify-content:space-between; align-items:center; gap:12px }
        .about-btn { background:linear-gradient(90deg,var(--accent),#ff7b7b); padding:10px 14px; border-radius:10px; color:#fff; font-weight:700; border:none }
        .search-box { width:100%; padding:10px 12px; border-radius:10px; border:1px solid rgba(255,255,255,0.04); background:transparent; color:#fff }
        .muted{ color:var(--muted) }
        /* clickable poster anchor resets */
        a.poster-link{ text-decoration:none }
        @media (max-width:768px){ .poster{height:200px} }
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

    # normalize some fields
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
        # update session state safely
        st.session_state['selected_movie'] = val


# --- Main App ---

def main():
    st.set_page_config(page_title='CineMatch', page_icon='🎬', layout='wide')
    inject_css()

    movies, similarity, all_genres = load_data()

    # pick up movie from query param if user clicked poster link
    set_selected_movie_from_query()

    # Header
    st.markdown("""
    <div class='topbar'>
      <div>
        <div class='cine-title'>🎬 CineMatch</div>
        <div class='cine-sub muted'>Find your next favorite movie — genre & search driven</div>
      </div>
      <div>
        <button class='about-btn' id='aboutBtn'>About</button>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # About modal (simple inline reveal using st.expander for compatibility)
    with st.expander('About CineMatch', expanded=False):
        st.markdown('''
        CineMatch is a compact, fast content-based recommender using movie metadata (overview, genres, cast, crew).
        Posters are fetched from TMDB. This UI supports two search modes: 1) Free-text (title/cast) and 2) Genre browser.\
        Click a poster to open the movie details (deep-link), or use the Recommendations button below any poster to jump straight into related picks.
        ''')

    st.markdown('---')

    # TWO search options: 1) free-text search  2) genre browsing
    search_col, genre_col = st.columns([2,1])

    # --- Free-text search ---
    with search_col:
        st.subheader('Search by title or cast')
        query = st.text_input('Type a movie title, actor or keyword', '')
        if st.button('Search', key='free_search') and query.strip():
            q = query.strip().lower()
            mask = movies['title'].str.lower().str.contains(q) | movies['cast_display'].str.lower().str.contains(q) | movies['overview'].apply(lambda o: ' '.join(o).lower()).str.contains(q)
            results = movies[mask]
            st.markdown(f"**Found {len(results)} results**")
            # display grid of first 24
            cols = st.columns(4)
            for i, (_, row) in enumerate(results.head(24).iterrows()):
                col = cols[i % 4]
                poster = fetch_poster(row.movie_id)
                with col:
                    # clicking poster uses query param to set selection
                    link = f"?selected_movie={requests.utils.requote_uri(row.title)}"
                    st.markdown(f"<a class='poster-link' href='{link}'><div class='movie-card'><img class='poster' src='{poster}' /><div class='movie-title'>{row.title}</div><div class='rating'>{row.vote_average}</div></div></a>", unsafe_allow_html=True)
                    # Recommendations button replaces old View
                    if st.button('Recommendations', key=f'rec_s_{row.movie_id}'):
                        st.session_state['selected_movie'] = row.title
                        names, posters, ratings, overviews = recommend(row.title, movies, similarity)
                        st.session_state['cur_recs'] = {'names':names,'posters':posters,'ratings':ratings,'overviews':overviews}

    # --- Genre browsing ---
    with genre_col:
        st.subheader('Browse by genre')
        selected_genres = st.multiselect('Choose genres', options=all_genres, key='genres')
        if selected_genres:
            # show count and small chips
            st.markdown(f"<div class='muted'>Showing movies that match: {' • '.join(selected_genres)}</div>", unsafe_allow_html=True)
            mask = movies['genres'].apply(lambda gl: any(g in (gl or []) for g in selected_genres))
            genre_df = movies[mask]
            st.markdown(f"**{len(genre_df)} movies**")
            # grid (3 columns)
            cols = st.columns(3)
            for i, (_, row) in enumerate(genre_df.head(30).iterrows()):
                col = cols[i % 3]
                poster = fetch_poster(row.movie_id)
                with col:
                    link = f"?selected_movie={requests.utils.requote_uri(row.title)}"
                    st.markdown(f"<a class='poster-link' href='{link}'><div class='movie-card'><img class='poster' src='{poster}' /><div class='movie-title'>{row.title}</div><div class='rating'>{row.vote_average}</div></div></a>", unsafe_allow_html=True)
                    if st.button('Recommendations', key=f'rec_g_{row.movie_id}'):
                        st.session_state['selected_movie'] = row.title
                        names, posters, ratings, overviews = recommend(row.title, movies, similarity)
                        st.session_state['cur_recs'] = {'names':names,'posters':posters,'ratings':ratings,'overviews':overviews}

    st.markdown('---')

    # If a movie is selected, show full details + recommendations (clicking poster sets query param above)
    if st.session_state.get('selected_movie'):
        sel = st.session_state['selected_movie']
        m = movies[movies['title'] == sel]
        if not m.empty:
            m = m.iloc[0]
            st.markdown(f"## 🎥 {m.title}")
            c1, c2 = st.columns([1,2])
            with c1:
                poster = fetch_poster(m.movie_id)
                st.image(poster, width=260)
                st.markdown(f"<div class='chip'>Rating: ⭐ {m.vote_average}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chip'>Genres: {' , '.join(m.genres)}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('**Overview**')
                st.write(' '.join(m.overview))
                st.markdown('**Cast**')
                st.write(m.cast_display)
                if st.button('Show Recommendations for this movie'):
                    names, posters, ratings, overviews = recommend(m.title, movies, similarity)
                    st.session_state['cur_recs'] = {'names':names,'posters':posters,'ratings':ratings,'overviews':overviews}

    # Render recommendations if available in session
    if st.session_state.get('cur_recs'):
        recs = st.session_state['cur_recs']
        st.markdown('---')
        st.subheader('✨ Recommended for you')
        cols = st.columns(4)
        for i in range(len(recs['names'])):
            with cols[i % 4]:
                st.image(recs['posters'][i], width=160)
                st.markdown(f"**{recs['names'][i]}**")
                st.markdown(f"<div class='muted'>⭐ {recs['ratings'][i]}</div>", unsafe_allow_html=True)
                if st.button('View Details', key=f'detail_rec_{i}'):
                    st.session_state['selected_movie'] = recs['names'][i]

    # Footer / Development notes
    st.markdown('---')
    st.markdown('**Development Journey (short):** converted notebook -> Streamlit, added genre-first browsing + free-text search, poster deep-links, and parallel poster fetching for speed.')


if __name__ == '__main__':
    main()

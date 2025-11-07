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

# --- Styling inspired by CineMatch.html ---
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        html, body, .stApp { background: #0e0e0e; color: #fff; font-family: 'Poppins', sans-serif; }
        h1 { color: #E50914; text-align:center; font-size:2.5rem; margin-top:0 }
        .subtitle { color:#ccc; text-align:center; margin-bottom:30px; }
        .movie-card { background: linear-gradient(180deg, #1a1a1a, #0d0d0d); border: 1px solid #333; border-radius: 12px; padding: 8px; text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease; }
        .movie-card:hover { transform: scale(1.05); box-shadow: 0 0 20px #E5091444; }
        .movie-card img { width:100%; height:240px; object-fit:cover; border-radius:8px; }
        .movie-title { font-weight:600; font-size:0.95rem; margin-top:8px; color:#fff; }
        .rating-badge { background:#E50914; padding:4px 8px; border-radius:6px; font-size:0.8rem; color:#fff; font-weight:700; display:inline-block; margin-top:6px; }
        .about-button { background:linear-gradient(90deg,#E50914,#ff5555); color:#fff; font-weight:700; border-radius:8px; border:none; padding:10px 16px; cursor:pointer; }
        .about-modal { text-align:center; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Data loading ---
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, List[List[float]], List[str]]:
    import pickle, requests
    r = requests.get(PICKLE_MOVIES_URL)
    movies_dict = pickle.loads(r.content)
    movies = pd.DataFrame(movies_dict)

    r2 = requests.get(PICKLE_SIM_URL)
    similarity = pickle.loads(r2.content)

    all_genres = sorted({g for genre_list in movies['genres'] for g in (genre_list or [])})
    return movies, similarity, all_genres

# --- Poster fetch ---
def fetch_poster(movie_id: int) -> str:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    try:
        res = requests.get(url, headers=HEADERS)
        data = res.json()
        if data.get('posters'):
            return f"https://image.tmdb.org/t/p/w500/{data['posters'][0]['file_path']}"
    except Exception:
        pass
    return "https://via.placeholder.com/300x450.png?text=No+Poster"

# --- Recommend ---
def recommend(title: str, movies: pd.DataFrame, similarity) -> Tuple[List[str], List[str], List[float]]:
    idx = movies[movies['title'] == title].index[0]
    distances = similarity[idx]
    indices = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:9]
    ids = [movies.iloc[i[0]].movie_id for i in indices]
    names = [movies.iloc[i[0]].title for i in indices]
    ratings = [movies.iloc[i[0]].vote_average for i in indices]
    with concurrent.futures.ThreadPoolExecutor() as ex:
        posters = list(ex.map(fetch_poster, ids))
    return names, posters, ratings

# --- Main App ---
def main():
    st.set_page_config(page_title='CineMatch', page_icon='🎬', layout='wide')
    inject_css()

    movies, similarity, all_genres = load_data()

    st.markdown("""<h1>🎬 CineMatch</h1><div class='subtitle'>Find your next favorite movie by genre</div>""", unsafe_allow_html=True)

    # WOW Factor: glowing About button at top right
    col1, col2 = st.columns([5,1])
    with col2:
        if st.button('About', key='about_btn', help='Know more about CineMatch'):
            st.markdown("""<div class='about-modal'><h3>About CineMatch</h3><p>CineMatch uses content-based recommendations to suggest films you'll love. Built with ❤️ using Streamlit & TMDB data.</p></div>""", unsafe_allow_html=True)

    # Sidebar - genre select and random movie
    with st.sidebar:
        st.header('🎭 Select Genre')
        selected_genres = st.multiselect('Pick a genre', all_genres)
        if st.button('🎲 Surprise Me!'):
            pick = random.choice(movies['title'].tolist())
            st.session_state['selected_movie'] = pick

    # Genre browsing
    if selected_genres:
        mask = movies['genres'].apply(lambda g: any(genre in (g or []) for genre in selected_genres))
        genre_df = movies[mask]
        st.markdown(f"### Showing {len(genre_df)} movies in {', '.join(selected_genres)}")
        cols = st.columns(5)
        for i, (_, row) in enumerate(genre_df.head(25).iterrows()):
            col = cols[i % 5]
            with col:
                st.markdown(
                    f"""<div class='movie-card'>
                    <img src='{fetch_poster(row.movie_id)}' />
                    <div class='movie-title'>{row.title}</div>
                    <div class='rating-badge'>⭐ {row.vote_average}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button('View', key=f'v{i}'):
                    st.session_state['selected_movie'] = row.title

    # Movie details
    if st.session_state.get('selected_movie'):
        sel = st.session_state['selected_movie']
        m = movies[movies['title'] == sel].iloc[0]
        st.markdown(f"## 🎥 {m.title}")
        c1, c2 = st.columns([1,2])
        with c1:
            st.image(fetch_poster(m.movie_id), width=250)
            st.markdown(f"**Rating:** ⭐ {m.vote_average}")
        with c2:
            st.markdown(f"**Overview:** {' '.join(m.overview)}")
        if st.button('Show Similar Movies'):
            names, posters, ratings = recommend(m.title, movies, similarity)
            st.subheader('✨ Similar Picks')
            cols = st.columns(4)
            for i in range(len(names)):
                with cols[i % 4]:
                    st.image(posters[i], width=160)
                    st.markdown(f"**{names[i]}**<br>⭐ {ratings[i]}", unsafe_allow_html=True)

if __name__ == '__main__':
    main()

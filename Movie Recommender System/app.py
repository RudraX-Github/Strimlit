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

# --- Styling (CineMatch WOW Edition) ---
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        :root { --accent:#E50914; --accent2:#ff6b6b; }
        html, body, .stApp {
            background: radial-gradient(900px 600px at 10% 10%, rgba(229,9,20,0.06), transparent),
                        linear-gradient(180deg, #050505, #0b0b0b);
            color: #fff;
            font-family: 'Inter', sans-serif;
        }
        .cine-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            color: var(--accent);
            margin-bottom: 0;
            animation: glow 3s infinite;
        }
        @keyframes glow {
            0%, 100% { text-shadow: 0 0 8px var(--accent); }
            50% { text-shadow: 0 0 24px var(--accent2); }
        }
        .cine-sub {
            text-align: center;
            color: #ccc;
            margin-bottom: 30px;
        }
        .movie-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 12px;
            padding: 8px;
            transition: transform 0.25s, box-shadow 0.25s;
        }
        .movie-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.6);
        }
        .poster {
            width: 100%;
            height: 260px;
            object-fit: cover;
            border-radius: 8px;
        }
        .movie-title {
            font-weight: 700;
            margin-top: 8px;
            font-size: 0.95rem;
        }
        .rating {
            background: var(--accent);
            padding: 6px 8px;
            border-radius: 8px;
            color: #fff;
            font-weight: 700;
            display: inline-block;
            margin-top: 6px;
        }
        .loader-wrap { display:flex; justify-content:center; padding:18px; }
        .loader-ball {
            width: 22px; height: 22px; border-radius: 50%;
            background: linear-gradient(90deg, var(--accent), var(--accent2));
            animation: bounce 0.8s infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-18px); }
        }
        .rec-card { animation: fadeIn 0.4s ease both; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: none; }
        }
        .floating-popup {
            position: fixed;
            right: 20px;
            top: 100px;
            width: 380px;
            z-index: 9999;
        }
        .floating-inner {
            background: rgba(30,30,30,0.95);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        }
        button[kind="primary"] {
            background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
            border: none !important;
            color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Data loading ---
@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, List[List[float]], List[str]]:
    r = requests.get(PICKLE_MOVIES_URL, timeout=10)
    movies_dict = pickle.loads(r.content)
    movies = pd.DataFrame(movies_dict)
    r2 = requests.get(PICKLE_SIM_URL, timeout=10)
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
        if data.get("posters"):
            fp = data["posters"][0].get("file_path")
            if fp:
                return f"https://image.tmdb.org/t/p/w500{fp}"
    except Exception:
        pass
    return "https://via.placeholder.com/300x450.png?text=No+Poster"

# --- Recommend ---
def recommend(title: str, movies: pd.DataFrame, similarity):
    matches = movies[movies["title"] == title]
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

# --- Button Handlers ---
def handle_set_selected(title: str):
    st.session_state["selected_movie"] = title

def handle_recommend_click(title: str, movies, similarity):
    st.session_state["compute_recs_for"] = title

# --- Main App ---
def main():
    st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")
    inject_css()

    movies, similarity, all_genres = load_data()

    st.markdown("<div class='cine-title'>🎬 CineMatch</div><div class='cine-sub'>Find your next favorite movie by title, cast, or genre</div>", unsafe_allow_html=True)

    # Search & genre filters
    query = st.text_input("🔎 Search by title / cast / keyword")
    selected_genres = st.multiselect("🎭 Filter genres", options=all_genres)

    # Combine filters
    df = movies
    if query:
        q = query.strip().lower()
        mask = (
            df["title"].str.lower().str.contains(q)
            | df["cast_display"].str.lower().str.contains(q)
            | df["overview"].apply(lambda o: " ".join(o).lower()).str.contains(q)
        )
        df = df[mask]
    if selected_genres:
        df = df[df["genres"].apply(lambda gl: any(g in (gl or []) for g in selected_genres))]

    st.markdown(f"**Showing {len(df)} results**")

    # Grid display
    cols = st.columns(4)
    ids = df["movie_id"].tolist()[:48]
    with concurrent.futures.ThreadPoolExecutor() as ex:
        posters = list(ex.map(fetch_poster, ids))

    for i, (_, row) in enumerate(df.head(48).iterrows()):
        col = cols[i % 4]
        with col:
            poster = posters[i]
            link = f"?popup_movie={requests.utils.requote_uri(row.title)}"
            st.markdown(
                f"<a href='{link}'><div class='movie-card'>"
                f"<img class='poster' src='{poster}' />"
                f"<div class='movie-title'>{row.title}</div>"
                f"</div></a>",
                unsafe_allow_html=True,
            )
            rcol1, rcol2 = st.columns([3, 1])
            with rcol1:
                st.markdown(f"<div class='rating'>⭐ {row.vote_average}</div>", unsafe_allow_html=True)
            with rcol2:
                st.button("✨", key=f"rec_{row.movie_id}", on_click=handle_recommend_click, args=(row.title, movies, similarity))
            st.button("Details", key=f"detail_{row.movie_id}", on_click=handle_set_selected, args=(row.title,))

    # Compute recs with loader
    if st.session_state.get("compute_recs_for"):
        title = st.session_state["compute_recs_for"]
        loader = st.empty()
        loader.markdown("<div class='loader-wrap'><div class='loader-ball'></div></div>", unsafe_allow_html=True)
        names, posters_r, ratings_r, _ = recommend(title, movies, similarity)
        st.session_state["cur_recs"] = {"names": names, "posters": posters_r, "ratings": ratings_r}
        del st.session_state["compute_recs_for"]
        loader.empty()

    # Floating popup on poster click
    params = st.experimental_get_query_params()
    if "popup_movie" in params:
        title = params["popup_movie"][0]
        row = movies[movies["title"] == title].iloc[0]
        popup_html = f"""
        <div class='floating-popup'>
            <div class='floating-inner'>
                <div style='display:flex; gap:12px'>
                    <img src='{fetch_poster(row.movie_id)}' style='width:110px; height:160px; object-fit:cover; border-radius:8px'/>
                    <div style='flex:1'>
                        <div style='font-weight:800; font-size:1rem'>{row.title}</div>
                        <div style='color:#cfcfcf; margin-top:6px'>{' '.join(row.overview)[:220]}...</div>
                        <div style='margin-top:8px'>
                            <span class='rating'>⭐ {row.vote_average}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(popup_html, unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Close popup"):
                st.experimental_set_query_params()
        with col2:
            st.button("Recommendations", on_click=handle_recommend_click, args=(row.title, movies, similarity))

    # Selected movie section
    if st.session_state.get("selected_movie"):
        title = st.session_state["selected_movie"]
        row = movies[movies["title"] == title].iloc[0]
        st.markdown("---")
        st.subheader(f"🎥 {row.title}")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(fetch_poster(row.movie_id), width=260)
        with c2:
            st.markdown("**Overview**")
            st.write(" ".join(row.overview))
            st.markdown("**Cast**")
            st.write(row.cast_display)
            st.button("Show Recommendations", on_click=handle_recommend_click, args=(row.title, movies, similarity))

    # Recommendations display
    if st.session_state.get("cur_recs"):
        recs = st.session_state["cur_recs"]
        st.markdown("---")
        st.subheader("✨ Recommendations")
        cols = st.columns(4)
        for i in range(len(recs["names"])):
            with cols[i % 4]:
                st.markdown(
                    f"<div class='rec-card'><img src='{recs['posters'][i]}' width='160' "
                    f"style='border-radius:8px'/><div style='font-weight:700;margin-top:8px'>{recs['names'][i]}</div>"
                    f"<div class='muted'>⭐ {recs['ratings'][i]}</div></div>",
                    unsafe_allow_html=True,
                )
                st.button("View", key=f"view_{i}", on_click=handle_set_selected, args=(recs["names"][i],))

if __name__ == "__main__":
    main()

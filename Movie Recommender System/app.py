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

# --- FIX 1: Corrected URLs ---
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
        :root { 
            --accent:#E50914; 
            --accent2:#ff6b6b; 
            --grad: linear-gradient(90deg, var(--accent), var(--accent2));
        }
        html, body, .stApp {
            background: radial-gradient(900px 600px at 10% 10%, rgba(229,9,20,0.06), transparent),
                        linear-gradient(180deg, #050505, #0b0b0b);
            color: #fff;
            font-family: 'Inter', sans-serif;
        }
        
        /* --- FIX 8: Illusive & Colorful Title --- */
        .cine-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            /* Animated Gradient Text */
            background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent));
            background-size: 200% auto;
            color: transparent;
            background-clip: text;
            -webkit-background-clip: text;
            animation: glow 3s infinite, gradient-flow 6s infinite ease-in-out;
        }
        @keyframes glow {
            0%, 100% { text-shadow: 0 0 8px var(--accent); }
            50% { text-shadow: 0 0 24px var(--accent2); }
        }
        @keyframes gradient-flow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        /* --- End Fix 8 --- */

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
            min-height: 2.8em; /* Consistent height */
            line-height: 1.4em;
        }
        .rating {
            background: var(--accent);
            padding: 6px 8px;
            border-radius: 8px;
            color: #fff;
            font-weight: 700;
            display: inline-block;
            margin-top: 6px;
            font-size: 0.9rem; /* Added for consistency */
        }
        .loader-wrap { display:flex; justify-content:center; padding:18px; }
        .loader-ball {
            width: 22px; height: 22px; border-radius: 50%;
            background: var(--grad);
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

        /* --- FIX 4: Floating Card Style (using st.experimental_dialog) --- */
        /* This targets the Streamlit experimental_dialog component */
        /* --- FIX: Revert CSS to stDialog for older API --- */
        /* --- FIX: Target stExperimentalModal instead --- */
        /* --- FIX: DELETING all modal/dialog CSS. It's not working. --- */
        
        /* --- NEW: Details Panel Style --- */
        /* This is the container that will hold movie details */
        .details-panel {
            background: rgba(30,30,30,0.85);
            backdrop-filter: blur(12px); 
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
            /* Animation */
            animation: fadeIn 0.4s ease both;
        }
        /* --- End NEW --- */

        button[kind="primary"] {
            background: var(--grad) !important;
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
    # --- FIX 1: Corrected URL ---
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    try:
        res = requests.get(url, headers=HEADERS, timeout=6)
        res.raise_for_status()
        data = res.json()
        if data.get("posters"):
            fp = data["posters"][0].get("file_path")
            if fp:
                # --- FIX 1: Corrected URL ---
                return f"https://image.tmdb.org/t/p/w500{fp}"
    except Exception:
        pass
    # --- FIX 1: Corrected URL ---
    return "https://via.placeholder.com/300x450.png?text=No+Poster"

# --- Recommend ---
def recommend(title: str, movies: pd.DataFrame, similarity):
    matches = movies[movies["title"] == title]
    if matches.empty:
        return [], [], [], []
    idx = matches.index[0]
    distances = similarity[idx]
    
    # --- FIX 7: Get top 8 + 2 wildcards ---
    sorted_indices = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)
    top_8_indices = sorted_indices[1:9]  # 8 most similar
    bottom_2_indices = sorted_indices[-2:] # 2 least similar
    indices = top_8_indices + bottom_2_indices
    
    ids = [movies.iloc[i[0]].movie_id for i in indices]
    names = [movies.iloc[i[0]].title for i in indices]
    ratings = [movies.iloc[i[0]].vote_average for i in indices]
    overviews = [movies.iloc[i[0]].overview for i in indices]
    with concurrent.futures.ThreadPoolExecutor() as ex:
        posters = list(ex.map(fetch_poster, ids))
    return names, posters, ratings, overviews

# --- Button Handlers (FIX 5) ---
def handle_popup_create(title: str):
    st.session_state["popup_title"] = title

def handle_popup_close():
    if "popup_title" in st.session_state:
        del st.session_state["popup_title"]

def handle_recommend_click(title: str):
    st.session_state["compute_recs_for"] = title
    handle_popup_close() # Close popup when recs are shown

def handle_view_rec_click(title: str):
    # "View" button on recs will also open the popup
    st.session_state["popup_title"] = title 

# --- Main App ---
def main():
    st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")
    inject_css()

    try:
        movies, similarity, all_genres = load_data()
    except Exception as e:
        st.error(f"Failed to load movie data. Please check connection/URLs. Error: {e}")
        st.stop()


    st.markdown("<div class='cine-title'>🎬 CineMatch</div><div class='cine-sub'>Find your next favorite movie by title, cast, or genre</div>", unsafe_allow_html=True)

    # --- NEW: Details Panel Container ---
    # This container will be populated when a movie is selected
    details_container = st.container()

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

    # --- NEW: Populate Details Panel ---
    # We move the logic that *was* in the dialog here.
    # It now renders inside the `details_container` we defined above.
    if "popup_title" in st.session_state:
        title = st.session_state.get("popup_title")
        
        if movies[movies["title"] == title].empty:
            handle_popup_close()
            st.rerun()

        row = movies[movies["title"] == title].iloc[0]

        with details_container:
            # Apply the CSS class to this container
            st.markdown("<div class='details-panel'>", unsafe_allow_html=True)
            
            st.subheader(f"Details for {row.title}")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(fetch_poster(row.movie_id), use_container_width=True)
            with c2:
                # --- FIX 3: Rating is now shown in popup ---
                st.markdown(f"<span class='rating'>⭐ {round(row.vote_average, 1)}</span>", unsafe_allow_html=True)
                st.markdown("**Overview**")
                st.write(" ".join(row.overview))
                st.markdown("**Cast**")
                st.write(row.cast_display)

            st.markdown("---")
            b1, b2 = st.columns([3, 1])
            with b1:
                st.button("Close", on_click=handle_popup_close, use_container_width=True)
            with b2:
                st.button(
                    "✨ Recommend", 
                    on_click=handle_recommend_click, 
                    args=(row.title,), 
                    use_container_width=True,
                    type="primary"
                )
            
            # Close the CSS div
            st.markdown("</div>", unsafe_allow_html=True)

    # Grid display
    cols = st.columns(4)
    ids = df["movie_id"].tolist()[:48]
    posters = []
    with concurrent.futures.ThreadPoolExecutor() as ex:
        posters = list(ex.map(fetch_poster, ids))

    for i, (_, row) in enumerate(df.head(48).iterrows()):
        col = cols[i % 4]
        with col:
            poster = posters[i]
            
            # --- FIX 2 & 3: Card is now a container with a button ---
            with st.container():
                # Card is just a static markdown box
                st.markdown(
                    # --- FIX 1: Corrected HTML typo (class='poster') ---
                    # --- FIX: Added rating to main grid card ---
                    f"<div class='movie-card'>"
                    f"<img class='poster' src='{poster}' />"
                    f"<div class='movie-title'>{row.title}</div>"
                    f"<span class='rating'>⭐ {round(row.vote_average, 1)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # This button triggers the popup
                st.button(
                    "Details", 
                    key=f"detail_{row.movie_id}", 
                    on_click=handle_popup_create, 
                    args=(row.title,),
                    use_container_width=True
                )
            # --- FIX 2: Removed old rating and buttons ---
            st.write("") # Adds a little space

    # Compute recs with loader
    if st.session_state.get("compute_recs_for"):
        title = st.session_state["compute_recs_for"]
        loader = st.empty()
        loader.markdown("<div class='loader-wrap'><div class='loader-ball'></div></div>", unsafe_allow_html=True)
        names, posters_r, ratings_r, _ = recommend(title, movies, similarity)
        st.session_state["cur_recs"] = {"names": names, "posters": posters_r, "ratings": ratings_r}
        del st.session_state["compute_recs_for"]
        loader.empty()

    # --- FIX 3, 4, 5, 6: Floating popup logic (now uses st.experimental_dialog) ---
    # --- DELETED ---
    # All of the `if "popup_title" in st.session_state:` logic
    # that contained `st.dialog`/`st.modal` is now GONE.
    # It has been moved into the `details_container` above.
    
    # Recommendations display
    if st.session_state.get("cur_recs"):
        recs = st.session_state["cur_recs"]
        st.markdown("---")
        st.subheader("✨ Recommendations")
        st.info("Showing Top 8 Similar and 2 'Wildcard' (Least Similar) Picks")
        cols = st.columns(5) # Changed to 5 columns
        
        # Display all 10 recommendations
        for i in range(len(recs["names"])):
            with cols[i % 5]:
                st.markdown(
                    f"""
                    <div class='rec-card'>
                        <img src='{recs['posters'][i]}' width='100%' 
                             style='border-radius:8px; object-fit: cover; height: 240px;'/>
                        <div style='font-weight:700;margin-top:8px'>{recs['names'][i]}</div>
                        <div>⭐ {round(recs['ratings'][i], 1)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # "View" button now opens the popup for the rec
                st.button("View", key=f"view_{i}", on_click=handle_view_rec_click, args=(recs["names"][i],), use_container_width=True)

if __name__ == "__main__":
    main()
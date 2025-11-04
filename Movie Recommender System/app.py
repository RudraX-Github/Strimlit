import streamlit as st
import pickle
import pandas as pd
import requests
import ast
import random
import concurrent.futures
import re 

# --- 2. API Config ---
API_READ_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzYmQzMGMxZmQ1YTkwNzdkODNlZGU1NDRiNzE5MGEzMCIsIm5iZiI6MTc2MjE1MjU2NS4wODcwMDAxLCJzdWIiOiI2OTA4NTA3NTMxZTQzNThmNDEwODE4MzUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.npzY38JNcrTkUKFDZ41XiZs_CmZsSls3oU63vo8gIIo"
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_READ_ACCESS_TOKEN}"
}

# --- 3. Helper Function: Fetch Movie Poster ---
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status() 
        data = response.json()
        if data.get('posters') and len(data['posters']) > 0:
            file_path = data['posters'][0].get('file_path')
            if file_path:
                return f"https://image.tmdb.org/t/p/w500/{file_path}"
    except requests.exceptions.RequestException as e:
        print(f"API Error in fetch_poster (ID: {movie_id}): {e}")
    # --- FIX 1: Corrected the fallback URL ---
    return "https://via.placeholder.com/500x750.png?text=Poster+Not+Available"


# --- 4. Helper Function: Get Movie Details ---
def get_movie_details(movie_df, title):
    """Fetches all details for a selected movie from the DataFrame."""
    try:
        movie = movie_df[movie_df['title'] == title].iloc[0]
        poster_url = fetch_poster(movie.movie_id)
        
        details = {
            "id": movie.get('movie_id', 'N/A'),
            "title": movie.get('title', 'Title Unknown'),
            "overview": movie.get('overview', []), # Expecting a list of words
            "genres": movie.get('genres', []), # Expecting a list of names
            "rating": movie.get('vote_average', 0.0),
            "poster_url": poster_url,
            "cast": movie.get('cast', ['N/A']), # Expecting a list of names
            "director": movie.get('director', 'N/A') # Expecting a string
        }
        return details
    except (IndexError, AttributeError):
        st.error(f"Could not find details for '{title}'.")
        return None

# --- 5. Helper Function: Get Recommendations ---
def recommend(movie_title):
    """
    Finds the top 8 similar movies and 2 least similar movies.
    Fetches all data in parallel.
    """
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
        distances = similarity[movie_index]
        sorted_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
        top_8_indices = sorted_indices[1:9]
        bottom_2_indices = sorted_indices[-2:]
        movies_list_indices = top_8_indices + bottom_2_indices
        
        movie_ids, movie_names, movie_overviews, movie_ratings, movie_genres = [], [], [], [], []
        
        for i in movies_list_indices:
            idx = i[0]
            movie_ids.append(movies.iloc[idx].movie_id)
            movie_names.append(movies.iloc[idx].title)
            movie_overviews.append(movies.iloc[idx].get('overview', []))
            movie_ratings.append(movies.iloc[idx].get('vote_average', 0.0))
            movie_genres.append(movies.iloc[idx].get('genres', []))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            movie_posters = list(executor.map(fetch_poster, movie_ids))
            
        return movie_names, movie_posters, movie_overviews, movie_ratings, movie_genres
        
    except IndexError:
        st.error(f"Movie '{movie_title}' not found in the dataset. Please try another.")
        return [], [], [], [], []
    except Exception as e:
        print(f"Full Error in recommend: {e}")
        st.error("An unexpected error occurred while generating recommendations.")
        return [], [], [], [], []

# --- 6. Helper Function: Filter Movies by Genre (FIXED) ---
def get_filtered_movies(movies_df, selected_genres):
    """Filters the movie DataFrame based on selected genres."""
    if not selected_genres:
        return movies_df
    
    def has_all_genres(genre_list):
        return all(genre in genre_list for genre in selected_genres)
        
    return movies_df[movies_df['genres'].apply(has_all_genres)]

# --- 7. Helper Function: Process and Load Data (FIXED) ---
@st.cache_data
def load_data():
    """
    Fetches and processes data.
    Fixes spaceless names (e.g., 'SamWorthington' -> 'Sam Worthington').
    """
    
    def format_name(name_string):
        if isinstance(name_string, str):
            return re.sub(r'([a-z])([A-Z])', r'\1 \2', name_string)
        return name_string
    
    movies_dict_url = 'https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/Movie%20Recommender%20System/pickle%20files/movies_dict.pkl'
    similarity_url = 'https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/Movie%20Recommender%20System/pickle%20files/similarity.pkl'
    
    try:
        response_dict = requests.get(movies_dict_url)
        response_dict.raise_for_status()
        movies_dict = pickle.loads(response_dict.content)
        movies = pd.DataFrame(movies_dict)
        
        required_cols = ['movie_id', 'title', 'overview', 'genres', 'vote_average', 'cast', 'crew']
        
        if not all(col in movies.columns for col in required_cols):
            st.error(
                "Data Error: Your 'movies_dict.pkl' file is missing columns. "
                f"Please ensure your notebook saves: {required_cols}"
            )
            st.stop()

        def extract_director_from_list(crew_list):
            if isinstance(crew_list, list) and len(crew_list) > 0:
                return crew_list[0]
            return 'N/A'
            
        movies['director'] = movies['crew'].apply(extract_director_from_list).apply(format_name)
        
        movies['cast'] = movies['cast'].apply(
            lambda names: [format_name(name) for name in names if isinstance(names, list)]
        )

        all_genres = sorted(list(set(g for genre_list in movies['genres'] for g in genre_list if isinstance(genre_list, list))))
        
        response_sim = requests.get(similarity_url)
        response_sim.raise_for_status()
        similarity = pickle.loads(response_sim.content)
        
        return movies, similarity, all_genres

    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading pickle files from GitHub: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.stop()

# --- 8. Helper Function: Set selected movie (for callbacks) ---
def set_selected_movie(title):
    """Callback function to update session state."""
    st.session_state.selected_movie = title

# ==========================================================
#                      LOAD DATA
# ==========================================================

try:
    with st.spinner('Loading movie data... This may take a moment on first load.'):
        movies, similarity, all_genres = load_data()
except Exception:
    st.error("A critical error occurred while loading data. The app cannot continue.")
    st.stop()

# ==========================================================
#                 9. STREAMLIT UI & CSS
# ==========================================================

st.set_page_config(
    page_title="CineMatch Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 3D CSS & STYLING OVERHAUL 🎨 ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
:root {
    --font-main: 'Montserrat', sans-serif;
    --color-bg: #111111;
    --color-content-bg: #181818;
    --color-border: #333333;
    --color-primary: #E50914;
    --color-primary-hover: #F61A25;
    --color-text-primary: #FFFFFF;
    --color-text-secondary: #AAAAAA;
    --color-shadow: rgba(0, 0, 0, 0.5);
}
html, body, .stApp {
    font-family: var(--font-main);
    background-color: var(--color-bg);
    color: var(--color-text-primary);
}
#MainMenu, footer { display: none; }
h1 {
    font-family: var(--font-main);
    font-weight: 700;
    font-size: 2.75rem;
    color: var(--color-primary);
    text-align: center;
    padding-bottom: 20px;
}
h2 {
    font-family: var(--font-main);
    font-weight: 600;
    font-size: 1.75rem;
    color: var(--color-text-primary);
    border-bottom: 2px solid var(--color-border);
    padding-bottom: 10px;
}
h3 {
    font-family: var(--font-main);
    font-weight: 600;
    color: var(--color-text-primary);
    font-size: 1.25rem;
}
.stSidebar > div:first-child {
    background: linear-gradient(180deg, #1E1E1E 0%, #111111 100%);
    border-right: 1px solid var(--color-border);
}
.stSidebar h2 {
    font-size: 1.5rem;
    border-bottom: 2px solid var(--color-primary);
    padding-bottom: 8px;
    margin-top: 10px;
}
.stAlert, .stInfo, .stWarning {
    background-color: #262626;
    border: 1px solid var(--color-border);
    border-left: 5px solid var(--color-primary);
    border-radius: 8px;
    color: var(--color-text-secondary);
}
.stButton > button[kind="primary"] {
    background-color: var(--color-primary);
    border: none;
    border-radius: 8px;
    padding: 14px 24px;
    font-size: 1.1rem;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(229, 9, 20, 0.2);
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--color-primary-hover);
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 25px rgba(229, 9, 20, 0.4);
}
.stButton > button:not([kind="primary"]) {
    background-color: var(--color-content-bg);
    color: var(--color-text-primary);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--color-primary);
    color: var(--color-primary);
    transform: translateY(-2px);
}
.stSelectbox > div > div {
    background-color: var(--color-content-bg);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    font-size: 1.05rem;
}
.stExpander {
    background-color: #262626;
    border: 1px solid var(--color-border);
    border-radius: 8px;
}
div[data-testid="stTabs"] button[role="tab"] {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--color-text-secondary);
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--color-primary);
    border-bottom: 3px solid var(--color-primary);
}
.selected-movie-box {
    background: linear-gradient(145deg, #1E1E1E, #242424);
    border-radius: 12px;
    padding: 24px 32px;
    border: 1px solid var(--color-border);
    box-shadow: 0 12px 32px var(--color-shadow);
    margin-bottom: 20px;
}
.selected-movie-box img {
    border-radius: 8px;
    box-shadow: 0 8px 24px var(--color-shadow);
}
.selected-movie-box h3 {
    color: var(--color-text-secondary);
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
    margin-top: 15px;
    font-weight: 600;
}
.selected-movie-box .stSubheader {
    font-size: 2rem;
    color: var(--color-text-primary);
    font-weight: 700;
}
.detail-list {
    font-size: 1rem;
    color: var(--color-text-primary);
    line-height: 1.6;
}
.movie-card {
    background-color: var(--color-content-bg);
    border: 1px solid var(--color-border);
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    height: 100%; 
    display: flex;
    flex-direction: column;
    position: relative;
    transform: perspective(1000px);
    transition: all 0.3s ease-in-out;
}
.movie-card:hover {
    transform: perspective(1000px) rotateY(-8deg) rotateX(4deg) scale(1.05);
    box-shadow: 0 12px 24px rgba(0,0,0,0.5);
    border-color: var(--color-primary);
}
.movie-card img {
    border-radius: 7px;
    width: 100%;
    max-height: 320px; 
    object-fit: cover;
    margin-bottom: 10px;
}
.movie-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--color-text-primary);
    min-height: 2.4em;
    line-height: 1.2em;
    margin-bottom: 10px;
}
.movie-rating {
    position: absolute;
    top: 18px;
    left: 18px;
    background-color: rgba(0, 0, 0, 0.85);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 5px 9px;
    border-radius: 8px;
    border: 1px solid #444;
}
.card-content {
    flex-grow: 1; 
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 15px;
    justify-content: flex-start;
}
.movie-card .tags-container { justify-content: center; }
.tag {
    background-color: var(--color-primary);
    color: var(--color-text-primary);
    padding: 5px 12px;
    border-radius: 15px;
    font-size: 0.75rem;
    font-weight: 600;
}
.selected-movie-box .tag {
    background-color: #333333;
    color: var(--color-text-secondary);
    font-weight: 600;
    font-size: 0.85rem;
    padding: 6px 14px;
}
.app-footer-info {
    text-align: center;
    color: var(--color-text-secondary);
    font-size: 0.8rem;
    border-top: 1px solid var(--color-border);
    padding-top: 20px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
#                   10. SIDEBAR CONTENT
# ==========================================================

with st.sidebar:
    st.header("🔍 Filter Movies")
    st.markdown("Select genres to find movies you're in the mood for.")
    selected_genres = st.multiselect(
        "Filter by genre:",
        options=all_genres,
        label_visibility="collapsed"
    )
    
# ==========================================================
#                   11. MAIN PAGE
# ==========================================================

st.title('CineMatch Movie Recommender')
st.markdown("---")
st.header("Choose a Movie You Like")

# --- Initialize Session State ---
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# --- Main Selection Area ---
col1, col2 = st.columns([3, 1])
with col1:
    def on_select_change():
        # only set selected_movie when user picks a real movie from the current filtered list
        val = st.session_state.movie_selector
        if val in filtered_movie_titles:
            st.session_state.selected_movie = val
        else:
            st.session_state.selected_movie = None

    # Ensure we build the filtered movie list based on the current genre filter
    filtered_movies = get_filtered_movies(movies, selected_genres)
    filtered_movie_titles = sorted(filtered_movies['title'].tolist())

    # Provide a harmless placeholder option when no movies match the filter
    options_list = filtered_movie_titles if filtered_movie_titles else ["(No movies match filter)"]

    try:
        current_index = filtered_movie_titles.index(st.session_state.selected_movie) if st.session_state.selected_movie in filtered_movie_titles else 0
    except Exception:
        current_index = 0 if options_list else None

    st.selectbox(
        "Type or select a movie from the dropdown:",
        options_list,
        index=current_index if isinstance(current_index, int) else None,
        placeholder="Select a movie from the list...",
        key='movie_selector',
        on_change=on_select_change
    )
with col2:
    def set_random_movie():
        if filtered_movie_titles:
            st.session_state.selected_movie = random.choice(filtered_movie_titles)
        else:
            st.warning("No movies found for the selected genre filter.")
    st.button(
        "🎲 Surprise Me!", 
        on_click=set_random_movie, 
        use_container_width=True,
        help="Pick a random movie from the filtered list"
    )

# --- Display Selected Movie's Details (FIXED) ---
if st.session_state.selected_movie:
    movie_details = get_movie_details(movies, st.session_state.selected_movie)
    
    if movie_details:
        st.markdown("---")
        st.header(f"You selected: {movie_details['title']}")
        
        st.markdown('<div class="selected-movie-box">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2], gap="medium")
        
        with col1:
            # --- FIX 2: Replaced use_container_width=True with width='stretch' ---
            st.image(movie_details['poster_url'], width='stretch')
        
        with col2:
            genre_html = "".join([f'<span class="tag">{g}</span>' for g in movie_details['genres']])
            st.markdown(f'<div class="tags-container">{genre_html}</div>', unsafe_allow_html=True)
            
            st.subheader(f"⭐ {movie_details['rating']:.1f} / 10")
            
            st.markdown("<h3>Overview</h3>", unsafe_allow_html=True)
            st.write(" ".join(movie_details['overview']))
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<h3>Cast</h3>", unsafe_allow_html=True)
                cast_list = ", ".join(movie_details['cast'])
                st.markdown(f'<div class="detail-list">{cast_list}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("<h3>Director</h3>", unsafe_allow_html=True)
                st.markdown(f'<div class="detail-list">{movie_details["director"]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # --- Main "Recommend" Button ---
        if st.button('Show Recommendations', use_container_width=True, type="primary"):
            st.header(f"Movies You Might Also Like")
            
            with st.spinner('Finding similar movies and fetching posters...'):
                names, posters, overviews, ratings, genres_lists = recommend(st.session_state.selected_movie)
                
                if names:
                    tab1, tab2 = st.tabs(["Top 5 Recommendations", "More to Explore (6-10)"])

                    with tab1:
                        cols_row1 = st.columns(5, gap="medium")
                        for i in range(5):
                            with cols_row1[i]:
                                st.markdown(f"""
                                <div class="movie-card">
                                    <div class="movie-rating">⭐ {ratings[i]:.1f}</div>
                                    <img src="{posters[i]}" alt="Poster for {names[i]}">
                                    <div class="card-content">
                                        <div class="movie-title">{names[i]}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.expander("Details"):
                                    genre_html = "".join([f'<span class="tag">{g}</span>' for g in genres_lists[i]])
                                    st.markdown(f'<div class="tags-container">{genre_html}</div>', unsafe_allow_html=True)
                                    st.write(f"**📖 Overview:** {" ".join(overviews[i])}")

                                st.button(
                                    "Find movies like this", 
                                    key=f"btn_rec_1_{i}",
                                    on_click=set_selected_movie,
                                    args=(names[i],),
                                    use_container_width=True
                                )
                    
                    with tab2:
                        st.info("Includes recommendations 6-8 and two 'wildcard' (least similar) picks.")
                        
                        cols_row2 = st.columns(5, gap="medium")
                        for i in range(5, 10):
                            with cols_row2[i-5]:
                                st.markdown(f"""
                                <div class="movie-card">
                                    <div class="movie-rating">⭐ {ratings[i]:.1f}</div>
                                    <img src="{posters[i]}" alt="Poster for {names[i]}">
                                    <div class="card-content">
                                        <div class="movie-title">{names[i]}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.expander("Details"):
                                    genre_html = "".join([f'<span class="tag">{g}</span>' for g in genres_lists[i]])
                                    st.markdown(f'<div class="tags-container">{genre_html}</div>', unsafe_allow_html=True)
                                    st.write(f"**📖 Overview:** {" ".join(overviews[i])}")

                                st.button(
                                    "Find movies like this", 
                                    key=f"btn_rec_2_{i}",
                                    on_click=set_selected_movie,
                                    args=(names[i],),
                                    use_container_width=True
                                )
                else:
                    st.error("Could not find any recommendations.")
    
elif not st.session_state.selected_movie and selected_genres:
    st.info("Select a movie from the filtered list to get started.")
else:
    st.info("Select a movie from the dropdown to get started.")

# --- "About" Footer ADDED ---
st.markdown("---")
st.markdown(
    """
    <p class="app-footer-info">
    This app recommends movies based on content similarity. The model analyzes movie tags (overview, genres, keywords, cast, and crew) 
    to find the 10 movies that are most similar to your selection.
    </p>
    """, 
    unsafe_allow_html=True
)
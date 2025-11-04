import streamlit as st
import pickle
import pandas as pd
import requests
import ast
import random
import concurrent.futures

# --- API Config ---
API_READ_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzYmQzMGMxZmQ1YTkwNzdkODNlZGU1NDRiNzE5MGEzMCIsIm5iZiI6MTc2MjE1MjU2NS4wODcwMDAxLCJzdWIiOiI2OTA4NTA3NTMxZTQzNThmNDEwODE4MzUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.npzY38JNcrTkUKFDZ41XiZs_CmZsSls3oU63vo8gIIo"
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_READ_ACCESS_TOKEN}"
}

# --- 1. Helper Function: Fetch Movie Poster ---
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
    return "https.via.placeholder.com/500x750.png?text=Poster+Not+Available"


# --- 2. Helper Function: Get Movie Details ---
def get_movie_details(movie_df, title):
    """Fetches all details for a selected movie from the DataFrame."""
    try:
        movie = movie_df[movie_df['title'] == title].iloc[0]
        poster_url = fetch_poster(movie.movie_id)
        
        details = {
            "id": movie.get('movie_id', 'N/A'),
            "title": movie.get('title', 'Title Unknown'),
            "overview": movie.get('overview', 'No overview available.'),
            "genres": movie.get('genres', []),
            "rating": movie.get('vote_average', 0.0),
            "poster_url": poster_url
        }
        return details
    except (IndexError, AttributeError):
        st.error(f"Could not find details for '{title}'.")
        return None

# --- 3. Helper Function: Get Recommendations ---
def recommend(movie_title):
    """
    Finds the top 10 similar movies and fetches all data in parallel.
    """
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
        distances = similarity[movie_index]
        movies_list_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
        
        movie_ids = []
        movie_names = []
        movie_overviews = []
        movie_ratings = []
        movie_genres = []
        
        for i in movies_list_indices:
            idx = i[0]
            movie_ids.append(movies.iloc[idx].movie_id)
            movie_names.append(movies.iloc[idx].title)
            movie_overviews.append(movies.iloc[idx].get('overview', 'No overview.'))
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

# --- 4. Helper Function: Process and Load Data ---
@st.cache_data
def load_data():
    """
    Fetches and processes data.
    """
    movies_dict_url = 'https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/Movie%20Recommender%20System/pickle%20files/movies_dict.pkl'
    similarity_url = 'https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/Movie%20Recommender%20System/pickle%20files/similarity.pkl'
    
    try:
        response_dict = requests.get(movies_dict_url)
        response_dict.raise_for_status()
        movies_dict = pickle.loads(response_dict.content)
        movies = pd.DataFrame(movies_dict)
        
        required_cols = ['movie_id', 'title', 'overview', 'genres', 'vote_average']
        if not all(col in movies.columns for col in required_cols):
            st.error(
                "Data Error: Your 'movies_dict.pkl' file is missing one or more "
                f"required columns: {required_cols}. "
            )
            st.stop()

        try:
            movies['genres'] = movies['genres'].apply(
                lambda x: [d['name'] for d in ast.literal_eval(x)] if isinstance(x, str) else []
            )
        except (ValueError, SyntaxError) as e:
            st.error(f"Data Error: Could not parse the 'genres' column: {e}")
            st.stop()

        all_genres = sorted(list(set(g for genre_list in movies['genres'] for g in genre_list)))
        
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

# --- 5. Helper Function: Filter Movies by Genre ---
def get_filtered_movies(movies_df, selected_genres):
    """Filters the movie DataFrame based on selected genres."""
    if not selected_genres:
        return movies_df
    
    def has_all_genres(genre_list):
        return all(genre in genre_list for genre in selected_genres)
        
    return movies_df[movies_df['genres'].apply(has_all_genres)]

# --- 6. Helper Function: Set selected movie (for callbacks) ---
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
#                      STREAMLIT UI
# ==========================================================

st.set_page_config(
    page_title="CineMatch Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 ADVANCED CSS INJECTION 🎨 ---
# This is a complete theme overhaul.
st.markdown("""
<style>
/* --- 1. Import Google Font --- */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

/* --- 2. Define CSS Variables (Theme) --- */
:root {
    --font-main: 'Montserrat', sans-serif;
    --color-bg: #111111; /* Dark background */
    --color-content-bg: #1E1E1E; /* Lighter background for cards */
    --color-border: #333333; /* Borders */
    --color-primary: #E50914; /* Netflix Red */
    --color-primary-hover: #F61A25; /* Brighter red for hover */
    --color-text-primary: #FFFFFF; /* White text */
    --color-text-secondary: #AAAAAA; /* Gray text */
    --color-shadow: rgba(0, 0, 0, 0.4);
}

/* --- 3. Global Styles --- */
html, body, .stApp {
    font-family: var(--font-main);
    background-color: var(--color-bg);
    color: var(--color-text-primary);
}

/* Hide Streamlit's default header and footer */
#MainMenu, footer {
    display: none;
}

/* Main title */
h1 {
    font-family: var(--font-main);
    font-weight: 700;
    color: var(--color-primary);
    text-align: center;
    padding-bottom: 20px;
}

/* Subheaders */
h2, h3 {
    font-family: var(--font-main);
    font-weight: 600;
    color: var(--color-text-primary);
}

/* --- 4. Streamlit Component Styling --- */

/* Sidebar */
.stSidebar > div:first-child {
    background-color: var(--color-content-bg);
    border-right: 1px solid var(--color-border);
}
.stSidebar .stImage img {
    border-radius: 10px;
    box-shadow: 0 4px 15px var(--color-shadow);
}
.stSidebar .stAlert, .stInfo {
    background-color: #262626;
    border: 1px solid var(--color-border);
    border-radius: 8px;
    color: var(--color-text-secondary);
}

/* Main "Show Recommendations" Button */
.stButton > button[kind="primary"] {
    background-color: var(--color-primary);
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    font-weight: 600;
    font-family: var(--font-main);
    transition: all 0.3s ease;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--color-primary-hover);
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(229, 9, 20, 0.3);
}

/* "Surprise Me" and "Find like this" Buttons */
.stButton > button {
    background-color: var(--color-content-bg);
    color: var(--color-text-primary);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    font-weight: 600;
    font-family: var(--font-main);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    border-color: var(--color-primary);
    color: var(--color-primary);
    transform: translateY(-2px);
}
.stButton > button:focus {
    border-color: var(--color-primary) !important;
    box-shadow: 0 0 0 2px rgba(229, 9, 20, 0.5) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: var(--color-content-bg);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    color: var(--color-text-primary);
}
/* Style dropdown menu */
div[data-baseweb="popover"] ul li {
    background-color: var(--color-content-bg);
    color: var(--color-text-primary);
}
div[data-baseweb="popover"] ul li:hover {
    background-color: #333333;
}

/* Expander */
.stExpander {
    background-color: #262626;
    border: 1px solid var(--color-border);
    border-radius: 8px;
}
.stExpander header {
    color: var(--color-text-secondary);
    font-weight: 600;
}
.stExpander header:hover {
    color: var(--color-primary);
}

/* Spinner */
.stSpinner > div > div {
    border-top-color: var(--color-primary);
}

/* Horizontal line */
hr {
    background: var(--color-border);
    height: 1px;
    border: none;
}

/* Tabs for Recommendations */
div[data-testid="stTabs"] button[role="tab"] {
    color: var(--color-text-secondary);
    font-weight: 600;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--color-primary);
    border-bottom: 2px solid var(--color-primary);
}

/* --- 5. Custom Card Components --- */

/* Selected Movie Box */
.selected-movie-box {
    background-color: var(--color-content-bg);
    border-radius: 12px;
    padding: 24px;
    border: 1px solid var(--color-border);
    box-shadow: 0 8px 24px var(--color-shadow);
}
.selected-movie-box img {
    border-radius: 8px;
    box-shadow: 0 4px 12px var(--color-shadow);
}
.selected-movie-box .stSubheader {
    color: var(--color-text-primary);
}
.selected-movie-box h3 { /* "Rating", "Overview" */
    color: var(--color-text-secondary);
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Recommendation Card */
.movie-card {
    background-color: var(--color-content-bg);
    border: 1px solid var(--color-border);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    height: 100%; 
    display: flex;
    flex-direction: column;
}
.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    border-color: var(--color-primary);
}
.movie-card img {
    border-radius: 7px;
    width: 100%;
    height: auto;
    object-fit: cover;
    margin-bottom: 10px;
}
.movie-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--color-text-primary);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    min-height: 2.4em;
    line-height: 1.2em;
    margin-bottom: 10px;
}
.card-content {
    flex-grow: 1; 
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* Genre Tags */
.tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 15px;
    justify-content: flex-start; /* Aligned left for selected movie */
}
.movie-card .tags-container {
    justify-content: center; /* Centered for recommendation cards */
}
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
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
#                   SIDEBAR CONTENT
# ==========================================================

with st.sidebar:
    st.image("https://image.tmdb.org/t/p/w500/qJ2tW6WMUDg92VazziDs16NtnLo.jpg", use_container_width=True)
    st.header("🎬 About CineMatch")
    st.info(
        "This app recommends movies based on content similarity. "
        "The model analyzes movie tags (overview, genres, keywords, cast, and crew) "
        "to find the 10 movies that are most similar to your selection."
    )
    
    st.header("⚙️ Model Details")
    st.markdown(
        """
        - **Data:** TMDB 5000 Movie Dataset
        - **Vectorization:** Bag-of-Words (CountVectorizer)
        - **Similarity:** Cosine Similarity
        """
    )
    
    st.header("🔍 Filter Movies")
    st.markdown("Select genres to narrow down the movie list below.")
    selected_genres = st.multiselect(
        "Filter by genre:",
        options=all_genres,
        label_visibility="collapsed"
    )

# Filter the main movie list
filtered_movies_df = get_filtered_movies(movies, selected_genres)
filtered_movie_titles = sorted(filtered_movies_df['title'].values)


# ==========================================================
#                     MAIN PAGE
# ==========================================================

st.title('CineMatch Movie Recommender')
st.markdown("---")
st.header("Step 1: Choose a Movie You Like")

# --- Initialize Session State ---
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# --- Main Selection Area ---
col1, col2 = st.columns([3, 1])

with col1:
    def on_select_change():
        st.session_state.selected_movie = st.session_state.movie_selector

    try:
        current_index = filtered_movie_titles.index(st.session_state.selected_movie)
    except ValueError:
        current_index = None

    st.selectbox(
        "Type or select a movie from the dropdown:",
        filtered_movie_titles,
        index=current_index,
        placeholder="Select a movie...",
        key='movie_selector',
        on_change=on_select_change
    )

with col2:
    def set_random_movie():
        if filtered_movie_titles:
            st.session_state.selected_movie = random.choice(filtered_movie_titles)
        else:
            st.warning("No movies match the selected genre filter.")

    st.button(
        "🎲 Surprise Me!", 
        on_click=set_random_movie, 
        use_container_width=True,
        help="Pick a random movie from the filtered list"
    )

# --- Display Selected Movie's Details ---
if st.session_state.selected_movie:
    movie_details = get_movie_details(movies, st.session_state.selected_movie)
    
    if movie_details:
        st.markdown("---")
        st.header(f"You selected: {movie_details['title']}")
        
        # --- NEW: Custom Container for Selected Movie ---
        # We replace st.container(border=True) with our custom CSS class
        st.markdown('<div class="selected-movie-box">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2], gap="medium")
        with col1:
            st.image(movie_details['poster_url'], use_container_width=True)
        with col2:
            genre_html = "".join([f'<span class="tag">{g}</span>' for g in movie_details['genres']])
            st.markdown(f'<div class="tags-container">{genre_html}</div>', unsafe_allow_html=True)
            
            st.subheader(f"⭐ {movie_details['rating']:.1f} / 10")
            
            st.markdown("<h3>Overview</h3>", unsafe_allow_html=True)
            st.write(movie_details['overview'])

        st.markdown('</div>', unsafe_allow_html=True)
        # --- End of Custom Container ---

        # --- Main "Recommend" Button ---
        if st.button('Show Recommendations', use_container_width=True, type="primary"):
            st.header(f"Step 2: Movies You Might Also Like")
            
            with st.spinner('Finding similar movies and fetching posters...'):
                names, posters, overviews, ratings, genres_lists = recommend(st.session_state.selected_movie)
                
                if names:
                    # --- NEW: Using st.tabs for a cleaner UI ---
                    tab1, tab2 = st.tabs(["Top 5 Recommendations", "You Might Also Like (6-10)"])

                    with tab1:
                        cols_row1 = st.columns(5, gap="medium")
                        for i in range(5):
                            with cols_row1[i]:
                                st.markdown(f"""
                                <div class="movie-card">
                                    <img src="{posters[i]}" alt="Poster for {names[i]}">
                                    <div class="card-content">
                                        <div class="movie-title">{names[i]}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.expander("Details"):
                                    genre_html = "".join([f'<span class="tag">{g}</span>' for g in genres_lists[i]])
                                    st.markdown(f'<div class="tags-container">{genre_html}</div>', unsafe_allow_html=True)
                                    st.write(f"**⭐ Rating:** {ratings[i]:.1f} / 10")
                                    st.write(f"**📖 Overview:** {overviews[i]}")

                                st.button(
                                    "Find movies like this", 
                                    key=f"btn_rec_1_{i}",
                                    on_click=set_selected_movie,
                                    args=(names[i],),
                                    use_container_width=True
                                )
                    
                    with tab2:
                        cols_row2 = st.columns(5, gap="medium")
                        for i in range(5, 10):
                            with cols_row2[i-5]:
                                st.markdown(f"""
                                <div class="movie-card">
                                    <img src="{posters[i]}" alt="Poster for {names[i]}">
                                    <div class="card-content">
                                        <div class="movie-title">{names[i]}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.expander("Details"):
                                    genre_html = "".join([f'<span class="tag">{g}</span>' for g in genres_lists[i]])
                                    st.markdown(f'<div class="tags-container">{genre_html}</div>', unsafe_allow_html=True)
                                    st.write(f"**⭐ Rating:** {ratings[i]:.1f} / 10")
                                    st.write(f"**📖 Overview:** {overviews[i]}")

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
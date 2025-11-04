import streamlit as st
import pickle
import pandas as pd
import requests
import ast  # For safely evaluating string representations of lists/dicts
import random
import concurrent.futures  # For parallel poster fetching

# The token you provided
API_READ_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzYmQzMGMxZmQ1YTkwNzdkODNlZGU1NDRiNzE5MGEzMCIsIm5iZiI6MTc2MjE1MjU2NS4wODcwMDAxLCJzdWIiOiI2OTA4NTA3NTMxZTQzNThmNDEwODE4MzUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.npzY38JNcrTkUKFDZ41XiZs_CmZsSls3oU63vo8gIIo"
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_READ_ACCESS_TOKEN}"
}

# --- 1. Helper Function: Fetch Movie Poster (No Changes) ---
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
    # Fallback placeholder
    return "https://via.placeholder.com/500x750.png?text=Poster+Not+Available"


# --- 2. Helper Function: Get Movie Details (for selected movie) ---
def get_movie_details(movie_df, title):
    """Fetches all details for a selected movie from the DataFrame."""
    try:
        movie = movie_df[movie_df['title'] == title].iloc[0]
        poster_url = fetch_poster(movie.movie_id)
        
        # Extract details, providing defaults if columns are missing
        details = {
            "id": movie.get('movie_id', 'N/A'),
            "title": movie.get('title', 'Title Unknown'),
            "overview": movie.get('overview', 'No overview available.'),
            "genres": movie.get('genres', []), # Assumes 'genres' is already a list
            "rating": movie.get('vote_average', 0.0),
            "poster_url": poster_url
        }
        return details
    except (IndexError, AttributeError):
        st.error(f"Could not find details for '{title}'.")
        return None

# --- 3. Helper Function: Get Recommendations (Updated) ---
def recommend(movie_title):
    """
    Finds the top 10 similar movies.
    Fetches all posters in parallel.
    Returns lists of names, posters, overviews, ratings, and genres.
    """
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
        distances = similarity[movie_index]
        # Get top 10 *similar* movies (indices 1 to 11)
        movies_list_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
        
        # --- Prepare data for all 10 movies ---
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

        # --- Fetch posters in parallel (Improvement #8) ---
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

# --- 4. Helper Function: Process and Load Data (Updated) ---
@st.cache_data
def load_data():
    """
    Fetches and processes data.
    IMPORTANT: This now assumes your 'movies_dict.pkl' contains
    'overview', 'genres', and 'vote_average' columns.
    """
    movies_dict_url = 'https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/Movie%20Recommender%20System/pickle%20files/movies_dict.pkl'
    similarity_url = 'https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/Movie%20Recommender%20System/pickle%20files/similarity.pkl'
    
    try:
        # Fetch and load movies_dict
        response_dict = requests.get(movies_dict_url)
        response_dict.raise_for_status()
        movies_dict = pickle.loads(response_dict.content)
        movies = pd.DataFrame(movies_dict)
        
        # --- Data Validation and Processing ---
        required_cols = ['movie_id', 'title', 'overview', 'genres', 'vote_average']
        if not all(col in movies.columns for col in required_cols):
            st.error(
                "Data Error: Your 'movies_dict.pkl' file is missing one or more "
                f"required columns: {required_cols}. "
                "Please regenerate your pickle file to include them."
            )
            st.stop()

        # Process 'genres' column
        # Assumes genres are stored as a string representation of a list of dicts
        # e.g., "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
        try:
            movies['genres'] = movies['genres'].apply(
                lambda x: [d['name'] for d in ast.literal_eval(x)] if isinstance(x, str) else []
            )
        except (ValueError, SyntaxError) as e:
            st.error(
                "Data Error: Could not parse the 'genres' column. "
                "Please ensure it's a string representation of a list of dicts (e.g., \"[{'name': 'Action'}]\")."
            )
            print(f"Genre parsing error: {e}")
            st.stop()

        # Get all unique genres for the filter (Improvement #6)
        all_genres = sorted(list(set(g for genre_list in movies['genres'] for g in genre_list)))
        
        # Fetch and load similarity
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
        return movies_df  # Return all movies if no genre is selected
    
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

# --- INJECT CUSTOM CSS (Updated) ---
st.markdown("""
<style>
/* --- Main Title --- */
h1 {
    color: #E50914; /* Netflix-like red */
    font-family: 'Arial', sans-serif;
    text-align: center;
    font-weight: bold;
}

/* --- Sidebar --- */
.stSidebar img {
    border-radius: 10px;
}

/* --- Selected Movie Details --- */
.selected-movie-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    border: 1px solid #ddd;
}
.selected-movie-card img {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* --- Recommendation Card (Updated for Flexbox) --- */
.movie-card {
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    
    /* --- Flexbox for equal height --- */
    height: 100%; 
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}
.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
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
    color: #111;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    min-height: 2.4em;
    line-height: 1.2em;
    margin-bottom: 10px;
}
/* This div wraps content below image to push button to bottom */
.card-content {
    flex-grow: 1; 
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* --- Genre Tags (Improvement #4) --- */
.tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
    justify-content: center; /* Center tags in the card */
}
.tag {
    background-color: #E50914;
    color: white;
    padding: 4px 10px;
    border-radius: 15px;
    font-size: 0.75rem;
    font-weight: 500;
}
.selected-movie-card .tag { /* Larger tags for selected movie */
    background-color: #c40812;
    padding: 6px 12px;
    font-size: 0.9rem;
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
    
    # --- Improvement #6: Genre Filter ---
    st.header("🔍 Filter Movies")
    st.markdown("Select genres to narrow down the movie list below.")
    selected_genres = st.multiselect(
        "Filter by genre:",
        options=all_genres,
        label_visibility="collapsed"
    )

# Filter the main movie list based on sidebar selection
filtered_movies_df = get_filtered_movies(movies, selected_genres)
filtered_movie_titles = sorted(filtered_movies_df['title'].values)


# ==========================================================
#                     MAIN PAGE
# ==========================================================

st.title('CineMatch Movie Recommender')
st.markdown("---")
st.header("Step 1: Choose a Movie You Like")

# --- Initialize Session State (Improvement #5) ---
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# --- Main Selection Area (Improvements #5 & #7) ---
col1, col2 = st.columns([3, 1])

with col1:
    # Update selected_movie in state if the selectbox is changed
    def on_select_change():
        st.session_state.selected_movie = st.session_state.movie_selector

    # Find the index of the currently selected movie to sync selectbox
    try:
        current_index = filtered_movie_titles.index(st.session_state.selected_movie)
    except ValueError:
        current_index = None

    st.selectbox(
        "Type or select a movie from the dropdown:",
        filtered_movie_titles,
        index=current_index,
        placeholder="Select a movie...",
        key='movie_selector',         # Key to link to session state
        on_change=on_select_change    # Callback to update state
    )

with col2:
    # --- Improvement #7: Random Movie Button ---
    def set_random_movie():
        st.session_state.selected_movie = random.choice(filtered_movie_titles)

    st.button(
        "🎲 Surprise Me!", 
        on_click=set_random_movie, 
        use_container_width=True,
        help="Pick a random movie from the filtered list"
    )

# --- Improvement #1: Display Selected Movie's Details ---
if st.session_state.selected_movie:
    movie_details = get_movie_details(movies, st.session_state.selected_movie)
    
    if movie_details:
        st.markdown("---")
        st.header(f"You selected: {movie_details['title']}")
        
        with st.container(border=True):
            col1, col2 = st.columns([1, 2], gap="medium")
            with col1:
                st.image(movie_details['poster_url'], use_container_width=True)
            with col2:
                # --- Improvement #4: Genre Tags ---
                genre_html = "".join([f'<span class="tag">{g}</span>' for g in movie_details['genres']])
                st.markdown(f'<div class="tags-container">{genre_html}</div>', unsafe_allow_html=True)
                
                st.subheader("⭐ Rating")
                st.subheader(f"{movie_details['rating']:.1f} / 10")
                
                st.subheader("📖 Overview")
                st.write(movie_details['overview'])

        # --- Main "Recommend" Button ---
        if st.button('Show Recommendations', use_container_width=True, type="primary"):
            st.header(f"Step 2: Movies You Might Also Like")
            
            with st.spinner('Finding similar movies and fetching posters...'):
                names, posters, overviews, ratings, genres_lists = recommend(st.session_state.selected_movie)
                
                if names:
                    # --- Improvement #9: 10 Recommendations in 2x5 Grid ---
                    
                    st.subheader("Top 5 Recommendations")
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
                            
                            # --- Improvement #3: Expander for Details ---
                            with st.expander("Details"):
                                # --- Improvement #2: Show More Details ---
                                # --- Improvement #4: Genre Tags ---
                                genre_html = "".join([f'<span class="tag">{g}</span>' for g in genres_lists[i]])
                                st.markdown(f'<div class="tags-container">{genre_html}</div>', unsafe_allow_html=True)
                                st.write(f"**⭐ Rating:** {ratings[i]:.1f} / 10")
                                st.write(f"**📖 Overview:** {overviews[i]}")

                            # --- Improvement #5: Click to Recommend ---
                            st.button(
                                "Find movies like this", 
                                key=f"btn_rec_1_{i}",
                                on_click=set_selected_movie,
                                args=(names[i],), # Pass the movie name to the callback
                                use_container_width=True
                            )

                    st.subheader("You Might Also Like")
                    cols_row2 = st.columns(5, gap="medium")
                    
                    for i in range(5, 10):
                        with cols_row2[i-5]: # Use i-5 for column index (0-4)
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
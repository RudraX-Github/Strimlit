import streamlit as st
import pickle
import pandas as pd
import requests
import time  # <-- 1. IMPORT TIME MODULE

# The token you provided
API_READ_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzYmQzMGMxZmQ1YTkwNzdkODNlZGU1NDRiNzE5MGEzMCIsIm5iZiI6MTc2MjE1MjU2NS4wODcwMDAxLCJzdWIiOiI2OTA4NTA3NTMxZTQzNThmNDEwODE4MzUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.npzY38JNcrTkUKFDZ41XiZs_CmZsSls3oU63vo8gIIo"

# --- Helper Function: Fetch Movie Poster (Using /images endpoint) ---
def fetch_poster(movie_id):
    """
    Fetches the movie poster URL from the TMDB API using the /images
    endpoint and an API Read Access Token (Bearer Token).
    """
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {API_READ_ACCESS_TOKEN}"
    }
    
    # Using the /images endpoint
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status() 
        data = response.json()
        
        # The /images endpoint returns a 'posters' list.
        # We need to check if the list is not empty and get the first one.
        if data.get('posters') and len(data['posters']) > 0:
            # Get the file path from the first poster
            file_path = data['posters'][0].get('file_path')
            if file_path:
                full_path = f"https://image.tmdb.org/t/p/w500/{file_path}"
                return full_path
            else:
                return "https://via.placeholder.com/500x750.png?text=No+Poster+Found"
        else:
            # No posters found in the list
            return "https://via.placeholder.com/500x750.png?text=No+Poster+Found"
            
    except requests.exceptions.RequestException as e:
        # --- MODIFIED ERROR HANDLING ---
        # Log the full error to the console (for your debugging)
        print(f"Full API Error in fetch_poster: {e}")
        # Show a simpler, user-friendly warning in the Streamlit app
        # We use st.warning since a missing poster isn't a critical failure
        st.warning(f"Could not fetch poster for one movie (ID: {movie_id}). Service may be unavailable.")
        return "https://via.placeholder.com/500x750.png?text=Poster+Error"

# --- Helper Function: Get Recommendations ---
def recommend(movie):
    """
    Finds the top 5 similar movies based on the pre-computed similarity matrix.
    All details (movie_id, title) come from the local .pkl file.
    Only the poster is fetched.
    """
    try:
        # Get data from local .pkl file
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommended_movie_names = []
        recommended_movie_posters = []
        
        for i in movies_list:
            # Get data from local .pkl file
            movie_id = movies.iloc[i[0]].movie_id
            movie_title = movies.iloc[i[0]].title
            
            # Fetch poster using the API
            recommended_movie_posters.append(fetch_poster(movie_id))
            recommended_movie_names.append(movie_title)
            
            # <-- 2. UPDATED DELAY -->
            # Add a small delay (500ms) between API calls to avoid rate limiting
            time.sleep(0.50)
            
        return recommended_movie_names, recommended_movie_posters
        
    except IndexError:
        st.error(f"Movie '{movie}' not found in the dataset. Please try another.")
        return [], []
    except Exception as e:
        # --- MODIFIED ERROR HANDLING ---
        # Log the full error to the console (for your debugging)
        print(f"Full Error in recommend: {e}")
         # Show a simpler, user-friendly error in the Streamlit app
        st.error("An unexpected error occurred while generating recommendations. Please try again.")
        return [], []

# --- Load Data (from your notebook's output) ---
try:
    # --- UPDATED FILE PATHS ---
    # Using raw strings (r'...') to correctly handle Windows paths
    movies_dict_path = r'D:\Git_HUB\Strimlit\Movie Recommender System\pickle files\movies_dict.pkl'
    similarity_path = r'D:\Git_HUB\Strimlit\Movie Recommender System\pickle files\similarity.pkl'
    
    movies_dict = pickle.load(open(movies_dict_path, 'rb'))
    movies = pd.DataFrame(movies_dict)
    
    similarity = pickle.load(open(similarity_path, 'rb'))
    
except FileNotFoundError:
    st.error("Error: One or more pickle files were not found.")
    st.info("Please check the file paths in app.py to ensure they are correct:")
    st.code(f"Movies Dict Path: {movies_dict_path}\nSimilarity Path: {similarity_path}", language="text")
    st.stop()
except Exception as e:
    st.error(f"Error loading pickle files: {e}")
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

# --- INJECT CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
/* --- Main Title --- */
h1 {
    color: #E50914; /* Netflix-like red */
    font-family: 'Arial', sans-serif;
    text-align: center;
    font-weight: bold;
}

/* --- Sidebar Image --- */
.stSidebar img {
    border-radius: 10px;
}

/* --- Movie Card Container --- */
.movie-card {
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    height: 100%; /* Important for column alignment */
    display: flex;
    flex-direction: column;
    justify-content: flex-start; /* Align content to top */
}
.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}

/* --- Movie Poster Image --- */
.movie-card img {
    border-radius: 7px;
    width: 100%;
    height: auto;
    object-fit: cover;
    margin-bottom: 10px; /* Space between image and title */
}

/* --- Movie Title Text --- */
.movie-title {
    font-size: 1rem; /* 16px */
    font-weight: 600; /* Semi-bold */
    color: #111;
    /* Clamp text to 2 lines to keep cards aligned */
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    min-height: 2.4em; /* Reserve space for 2 lines */
    line-height: 1.2em;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("https://image.tmdb.org/t/p/w500/qJ2tW6WMUDg92VazziDs16NtnLo.jpg", use_container_width=True)
    st.header("🎬 About CineMatch")
    st.info(
        "This app recommends movies based on content similarity. "
        "The model analyzes movie tags (overview, genres, keywords, cast, and crew) "
        "to find the 5 movies that are most similar to your selection."
    )
    st.header("⚙️ Model Details")
    st.markdown(
        """
        - **Data:** TMDB 5000 Movie Dataset
        - **Vectorization:** Bag-of-Words (CountVectorizer)
        - **Similarity:** Cosine Similarity
        """
    )

st.title('CineMatch Movie Recommender')
st.markdown("---")
st.header("Step 1: Choose a Movie You Like")

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown:",
    movies['title'].values,
    index=None,
    placeholder="Select a movie..."
)

if st.button('Show Recommendations', use_container_width=True, type="primary"):
    if selected_movie:
        st.header(f"Step 2: Movies You Might Also Like")
        
        # This spinner will run while the app tries to connect
        with st.spinner('Fetching recommendations and posters...'):
            names, posters = recommend(selected_movie)
        
            if names: 
                col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
                cols = [col1, col2, col3, col4, col5]
                
                # --- UPDATED DISPLAY LOOP ---
                # Use st.markdown to create styled "movie cards"
                for i in range(len(names)):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="movie-card">
                            <img src="{posters[i]}" alt="Poster for {names[i]}">
                            <div class="movie-title">{names[i]}</div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.warning("Please select a movie first.")


import streamlit as st
import pickle
import pandas as pd
import requests

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
    
    # *** THIS IS THE CHANGE YOU REQUESTED ***
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
        # You will still get the ConnectionResetError here
        st.error(f"API Error: {e}")
        return "https://via.placeholder.com/500x750.png?text=API+Error"

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
            
        return recommended_movie_names, recommended_movie_posters
        
    except IndexError:
        st.error(f"Movie '{movie}' not found in the dataset. Please try another.")
        return [], []
    except Exception as e:
        st.error(f"An error occurred during recommendation: {e}")
        return [], []

# --- Load Data (from your notebook's output) ---
try:
    movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'movies_dict.pkl' or 'similarity.pkl' not found.")
    st.info("Please run the 'Movie.ipynb' notebook to generate the required pickle files.")
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
        
        # This spinner will run while the app tries (and fails) to connect
        with st.spinner('Fetching recommendations and posters...'):
            names, posters = recommend(selected_movie)
        
            if names: 
                col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
                cols = [col1, col2, col3, col4, col5]
                
                for i in range(len(names)):
                    with cols[i]:
                        st.text(names[i])
                        st.image(posters[i], use_container_width=True, caption=f"Poster for {names[i]}")
    else:
        st.warning("Please select a movie first.")
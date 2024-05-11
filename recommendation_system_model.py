import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import coo_matrix
import streamlit as st

# Define HTML/CSS style with background image
html_style = """
<style>
    [data-testid="stAppViewContainer"]{
        background-image: url('https://images.unsplash.com/photo-1482855549413-2a6c9b1955a7?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-size: cover; 
        background-position: center;
        height: 100%;
    }
    h1{
    text-align:center;
    margin-bottom: 5px; 
    margin-top: -20px;
    }
    .main-title-2{
        font-family: 'Poetsen One', sans-serif;
        margin-bottom: 30px; 
        margin-top: 10px;
        text-align: center; 
        }
    
    .recommendation-header {
        font-family: 'Poetsen One', sans-serif;
        font-size: 15px;
    }
    .recommended-game {
        font-family: 'Poetsen One', sans-serif;
        font-size: 16px;
        margin-left: 20px; /* Add margin for better indentation */
    }
    .warning-text {
        font-family: 'Poetsen One', sans-serif;
        font-size: 16px;
        color: red;
        text-align: center;
    }
    .top-recommended-header {
        font-family: 'Poetsen One', sans-serif;
        font-size: 15px;   
    }
    p {
    font-size: 18px;
    }
</style>
"""

# Render HTML/CSS style
st.markdown(html_style, unsafe_allow_html=True)

# Function to reduce memory usage
def reduce_memory(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# Loading datasets
@st.cache_data
def load_data():
    games_df = reduce_memory(pd.read_csv('games.csv'))
    users_df = reduce_memory(pd.read_csv('users.csv'))
    recommendations_df = reduce_memory(pd.read_csv('recommendations.csv'))
    return games_df, users_df, recommendations_df

games_df, users_df, recommendations_df = load_data()

# Print out unique values in the 'title' column
print(games_df['title'].unique())

# Replace NaN values in the 'title' column with empty strings
cleaned_titles = games_df['title'].fillna('')

# Define the TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1, stop_words='english', dtype=np.float32)

# Fit and transform the tfidf vectorizer
tfidf_matrix = tf.fit_transform(cleaned_titles)

# Building user-user recommendation model
unique_user_ids = recommendations_df['user_id'].unique()
unique_app_ids = recommendations_df['app_id'].unique()

user_id_indices = {id: index for index, id in enumerate(unique_user_ids)}
app_id_indices = {id: index for index, id in enumerate(unique_app_ids)}

user_indices = [user_id_indices[user_id] for user_id in recommendations_df['user_id']]
app_indices = [app_id_indices[app_id] for app_id in recommendations_df['app_id']]
hours = recommendations_df['hours'].tolist()

user_user_matrix = coo_matrix((hours, (user_indices, app_indices)), shape=(len(unique_user_ids), len(unique_app_ids)))

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_user_matrix)

# Building game-game recommendation model
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

unique_game_ids = games_df['app_id'].astype('category').cat.categories

# Streamlit App
st.markdown("<h1 class='main-title-2'>Game Recommendation System</h1>", unsafe_allow_html=True)
st.write('\n')
st.write('\n')

# Function to get similar users
def get_similar_users(user_id, user_user_matrix, knn_model, n_neighbors=6):
    distances, indices = knn_model.kneighbors(user_user_matrix.getrow(user_id), n_neighbors=n_neighbors)
    similar_users = [unique_user_ids[i] for i in indices.flatten()[1:]]
    return similar_users


# Function to get similar games
def get_similar_games(game_id, tfidf, games_data, n_neighbors=6):
    game_index = np.where(unique_game_ids == game_id)[0]
    if len(game_index) == 0:
        return []
    game_index = game_index[0]
    
    # Fill NaN values in game titles with empty strings
    cleaned_titles = games_data['title'].fillna('')
    
    tfidf_matrix = tfidf.fit_transform(cleaned_titles)
    
    cosine_similarities = linear_kernel(tfidf_matrix[game_index], tfidf_matrix).flatten()
    cosine_similarities = cosine_similarities.squeeze()  # Squeeze to remove extra dimension
    similar_indices = cosine_similarities.argsort()[:-n_neighbors-1:-1]
    similar_games = [(cleaned_titles.iloc[i], cosine_similarities[i]) for i in similar_indices if i != game_index]
    return similar_games

def recommend_games(user_id):
    similar_users = get_similar_users(user_id, user_user_matrix, knn_model)
    similar_games = {}
    
    # Loop through similar users
    for user in similar_users:
        user_games = recommendations_df[recommendations_df['user_id'] == user]['app_id'].unique()
        
        # Loop through each game of the similar user
        for game_id in user_games:
            similar_games.setdefault(game_id, 0)
            
            # Get similar games for each game of the similar user
            for game, similarity in get_similar_games(game_id, tf, games_df):
                # Update the similarity score for the game
                similar_games[game] += similarity
    
    # Filter out the games already played by the user
    played_games = recommendations_df[recommendations_df['user_id'] == user_id]['app_id'].unique()
    similar_games = {game_id: score for game_id, score in similar_games.items() if game_id not in played_games}
    
    # Sort the games by similarity score and get top 5 recommendations
    sorted_games = sorted(similar_games.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Get game titles from game IDs
    recommended_games = []
    for game_id, score in sorted_games:
        game_title = games_df.loc[games_df['app_id'] == game_id, 'title'].iloc[0] if game_id in games_df['app_id'].values else "Game Title Not Found"
        recommended_games.append((game_title, score))
    
    return recommended_games






# Initialize session state
if 'recommended_games' not in st.session_state:
    st.session_state.recommended_games = []

# Recommending games for a specific user
user_id = st.number_input("Enter User ID:", min_value=1, max_value=users_df['user_id'].max(), value=1, step=1)

if st.button("Recommend Games"):

    recommended_games = recommend_games(user_id)
    if recommended_games:
        st.session_state.recommended_games = recommended_games
        st.markdown(f"<p class='top-recommendation-header'>Recommended Games for User ID {user_id}:</p>", unsafe_allow_html=True)
        for game, score in recommended_games:
            st.markdown(f"<p class='recommended-game'>• {game} (Similarity Score: {score:.2f})</p>", unsafe_allow_html=True)
    else:
        st.warning("No games found for recommendation.")

st.write('\n')
st.write('\n')

# Selecting a game from the recommended games list
recommended_game_titles = [game[0] for game in st.session_state.recommended_games] if st.session_state.recommended_games else []
selected_game_title = st.selectbox("Select a game to recommend 5 games for it:", recommended_game_titles, index=0, key="game_dropdown")

if selected_game_title:
    selected_game_index = recommended_game_titles.index(selected_game_title) if selected_game_title in recommended_game_titles else -1
    st.write(f"Selected Game Title: {selected_game_title}")
    st.write(f"Selected Game Index: {selected_game_index}")
    st.write(f"Recommended Game Titles: {recommended_game_titles}")
    if selected_game_index >= 0:
        st.write(f"<p class='top-recommendation-header'>Top 5 recommended games for {selected_game_title}:</p>", unsafe_allow_html=True)
        similar_games = recommend_games(selected_game_index)  # Pass selected_game_index instead of user_id
        for game_title, _ in similar_games[:5]:
            st.write(f"<p class='recommended-game'>• {game_title}</p>", unsafe_allow_html=True)
    else:
        st.warning("Selected game title not found.")



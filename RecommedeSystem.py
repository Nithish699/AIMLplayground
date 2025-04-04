# Import required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds  # Alternative for SVD

# Step 1: Load dataset
df = pd.read_csv('disney_plus_titles.csv')

# ------------------- Content-Based Filtering -------------------

# Step 2: Preprocess the text data (Handle missing values)
df['description'] = df['description'].fillna('')

# Step 3: Convert text to numerical vectors using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Step 4: Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Save the content-based model
pickle.dump(cosine_sim, open('RD_content_model.pkl', 'wb'))

# ------------------- Collaborative Filtering (Without surprise) -------------------

# Step 6: Create a synthetic user-item rating dataset
np.random.seed(42)
num_users = 1000  # Assume 1000 users
num_movies = df.shape[0]  # Number of movies

df_ratings = pd.DataFrame({
    'userId': np.random.randint(1, num_users, num_movies),
    'movieId': range(num_movies),
    'rating': np.random.randint(1, 6, num_movies)
})

# Step 7: Create user-item matrix
user_movie_matrix = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Step 8: Perform SVD using SciPy (Alternative to surprise)
U, sigma, Vt = svds(user_movie_matrix.values, k=50)  # Keep 50 latent features
sigma = np.diag(sigma)

# Step 9: Reconstruct the approximate ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Step 10: Save the collaborative filtering model
pickle.dump(predicted_ratings, open('RD_collaborative_model.pkl', 'wb'))

print("✅ Models saved: 'content_model.pkl' and 'collaborative_model.pkl'")

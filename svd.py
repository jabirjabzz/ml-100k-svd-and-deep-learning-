import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy  # Import accuracy from surprise.accuracy
import os

# List all files in the input directory
dirname, _, filenames in os.walk('/kaggle/input')

# Assuming the dataset is available in the /kaggle/input/ directory
# Load the data into pandas DataFrame
ratings_file = '/kaggle/input/ml-100k/ml-100k/u.data'
ratings = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Display the first few rows of the DataFrame
print(ratings.head())

# Create a Reader object for the Surprise library
reader = Reader(rating_scale=(1, 5))

# Load the data into the Surprise Dataset object
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm for collaborative filtering
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

# Predict ratings for the testset
predictions = algo.test(testset)

# Evaluate the performance
accuracy.rmse(predictions)

# Example of generating recommendations for a specific user
user_id = 196  # Replace with a user ID from the dataset
# Get a list of all movie IDs
all_movie_ids = ratings['item_id'].unique()

# Get a list of movies the user has already rated
user_rated_movies = ratings[ratings['user_id'] == user_id]['item_id'].unique()

# Find movies that the user has not rated
unrated_movies = [movie for movie in all_movie_ids if movie not in user_rated_movies]

# Predict ratings for the unrated movies
predictions = [algo.predict(user_id, movie_id) for movie_id in unrated_movies]

# Sort the predictions by estimated rating in descending order
top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

# Print the top 10 recommendations for the user
print("Top 10 movie recommendations for user ID:", user_id)
for prediction in top_predictions[:10]:
    print(f"Movie ID: {prediction.iid}, Estimated Rating: {prediction.est}")

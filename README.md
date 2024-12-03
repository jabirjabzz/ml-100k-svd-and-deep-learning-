Deep Learning-Based Movie Recommendation System
Overview
This repository contains a deep learning-based movie recommendation system built using the MovieLens dataset. The system leverages neural network embeddings to predict user ratings for movies and generate personalized recommendations.

Dataset
The dataset used is the MovieLens 100k dataset, which contains:

100,000 ratings from 943 users on 1,682 movies.

Each user has rated at least 20 movies.

Simple demographic information for the users (age, gender, occupation, zip code).

Project Structure
data/: Contains the MovieLens dataset files.

notebooks/: Jupyter notebooks for data exploration, model training, and evaluation.

scripts/: Python scripts for data preprocessing, model training, saving, and loading.

saved_models/: Directory to save trained models.

Requirements
Python 3.x

pandas

numpy

scikit-learn

tensorflow

You can install the required packages using pip:

bash
pip install pandas numpy scikit-learn tensorflow
Usage
Data Preparation
Load and preprocess the dataset:

python
import pandas as pd

ratings_file = 'data/ml-100k/u.data'
ratings = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
Model Training
Train the neural network model:

python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot

# Model architecture
embedding_size = 50
n_users = ratings['user_id'].nunique()
n_items = ratings['item_id'].nunique()

user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(n_users, embedding_size, name='user_embedding')(user_input)
user_embedding = Flatten(name='user_flatten')(user_embedding)

item_input = Input(shape=(1,), name='item_input')
item_embedding = Embedding(n_items, embedding_size, name='item_embedding')(item_input)
item_embedding = Flatten(name='item_flatten')(item_embedding)

dot_product = Dot(axes=1)([user_embedding, item_embedding])
model = Model(inputs=[user_input, item_input], outputs=dot_product)
model.compile(optimizer='adam', loss='mean_squared_error')

# Training
model.fit([train_user_ids, train_item_ids], train_ratings, epochs=10, validation_split=0.1)
model.save('saved_models/deep_learning_model.h5')
Model Evaluation
Load the trained model and evaluate its performance:

python
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the model
model = load_model('saved_models/deep_learning_model.h5')

# Evaluate
test_predictions = model.predict([test_user_ids, test_item_ids])
rmse = np.sqrt(mean_squared_error(test_ratings, test_predictions))
print(f'RMSE: {rmse}')
Generating Recommendations
Generate movie recommendations for a specific user:

python
user_id = 196  # Replace with any user ID
unrated_items = [item for item in range(n_items) if item not in train_item_ids[train_user_ids == user_id]]
predicted_ratings = [model.predict([np.array([user_id]), np.array([item_id])])[0][0] for item_id in unrated_items]
top_items = np.argsort(predicted_ratings)[-10:][::-1]

print("Top 10 movie recommendations for user ID:", user_id)
for item_id in top_items:
    print(f"Movie ID: {unrated_items[item_id]}, Predicted Rating: {predicted_ratings[item_id]}")
Acknowledgements
This project uses the MovieLens dataset collected by the GroupLens Research Project at the University of Minnesota.

License
This project is licensed under the MIT License. See the LICENSE file for details.

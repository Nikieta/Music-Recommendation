import streamlit as st
import pandas as pd
import numpy as np
import os
import Recommenders as Recommenders

# Change the directory and load the data
os.chdir('/Users/amardipgoswami/Downloads/PRJ Music Recommendation System/')
song_df_1 = pd.read_csv('triplets_file.csv')
song_df_2 = pd.read_csv('song_data.csv')

# Combine two data frames and create one data frame
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on='song_id', how='left')
song_df = song_df.head(50000)  # Select only 50000 records to create a model

# Create a new feature combining title and artist name
song_df['song'] = song_df['title'] + ' - ' + song_df['artist_name']

# Initialize the Streamlit app
st.title("Music Recommendation System")

# Create a sidebar
st.sidebar.title("Recommendation Options")

# User input for recommendation type
recommendation_type = st.sidebar.selectbox("Select Recommendation Type", ["Popularity", "Item Similarity"])

# User input for song name (optional)
song_name = st.sidebar.text_input("Enter a song name for recommendations")

# Initialize the recommenders
pr = Recommenders.popularity_recommender_py()
ir = Recommenders.item_similarity_recommender_py()

# Create the recommendation models
pr.create(song_df, 'user_id', 'song')
ir.create(song_df, 'user_id', 'song')

# Function to get recommendations based on song name
def get_song_recommendations(song_name):
    if song_name:
        return ir.get_similar_items([song_name])
    else:
        return None

# Function to get recommendations based on user ID
def get_user_recommendations(user_id):
    if recommendation_type == "Popularity":
        return pr.recommend(user_id)
    elif recommendation_type == "Item Similarity":
        return ir.recommend(user_id)
    else:
        return None

# Get the recommendations
if st.sidebar.button("Get Recommendations"):
    if song_name:
        recommendations = get_song_recommendations(song_name)
    else:
        recommendations = get_user_recommendations(user_id)

    # Display the recommendations
    st.write("Recommendations:")
    for i, recommendation in enumerate(recommendations):
        st.write(f"{i+1}. {recommendation}")

if __name__ == "__main__":
    st.run()
#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast

st.title("🎶 Hybrid Song Recommender System")

# Load the CSV files
user_song_file = "user_song_interactions.csv"
song_features_file = "song_features.csv"

# Load Data
user_song_df = pd.read_csv(user_song_file)
song_features_df = pd.read_csv(song_features_file)

st.subheader("📊 User-Song Interaction Data Preview")
st.write(user_song_df.head())

st.subheader("📊 Song Features Data Preview")
st.write(song_features_df.head())

# Ensure the user-song interactions file has the correct columns
if all(col in user_song_df.columns for col in ["user", "song", "rating"]):
    # Step 1: Collaborative Filtering using KNN
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_song_df[["user", "song", "rating"]], reader)

    # Step 2: Train/Test Split
    trainset, testset = train_test_split(data, test_size=0.2)

    # Step 3: Build Collaborative Filtering Model (KNN)
    algo = KNNBasic(sim_options={"user_based": True})
    algo.fit(trainset)

    # Step 4: Evaluate Collaborative Model
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    st.write(f"📉 Collaborative Filtering RMSE on test data: {rmse:.2f}")

    # Step 5: Full Training for Collaborative Filtering
    trainset_full = data.build_full_trainset()
    algo.fit(trainset_full)
    anti_testset = trainset_full.build_anti_testset()
    full_predictions = algo.test(anti_testset)

    # Step 6: Get top-N collaborative recommendations
    def get_top_n(predictions, n=5):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
        return top_n

    top_n_collab = get_top_n(full_predictions, n=5)

    # Step 7: Content-Based Filtering
    def get_content_based_recommendations(song_id, song_features_df, top_n=5):
        song = song_features_df[song_features_df['filename'] == song_id]
        if song.empty:
            return pd.DataFrame()
        try:
            feature_columns = ['tempo'] + [col for col in song_features_df.columns if col.startswith('mfcc') or col.startswith('chroma')]
            song_features = song[feature_columns].values
            feature_matrix = song_features_df[feature_columns].values
            # Clean song_features (single row)
            song_features_cleaned = np.array([
                [ast.literal_eval(str(x))[0] if isinstance(x, str) and x.startswith('[') else float(x) for x in row]
                for row in song_features
            ], dtype=float)

            # Clean feature_matrix (all songs)
            feature_matrix_cleaned = np.array([
                [ast.literal_eval(str(x))[0] if isinstance(x, str) and x.startswith('[') else float(x) for x in row]
                for row in feature_matrix
            ], dtype=float)
            cosine_similarities = cosine_similarity(song_features_cleaned, feature_matrix_cleaned)
            similar_songs = cosine_similarities.flatten()
            similar_indices = similar_songs.argsort()[-top_n-1:-1][::-1]
            return song_features_df.iloc[similar_indices]
        except KeyError:
            return pd.DataFrame()

    # Step 8: Combine Collaborative and Content-Based Recommendations
    def hybrid_recommendations(user_id, top_n_collab, song_features_df, alpha=0.5, num_recs=5):
       
        collab_recs = top_n_collab.get(user_id, [])
        hybrid_recs = []
        feature_columns = ['tempo'] + [col for col in song_features_df.columns if col.startswith('mfcc') or col.startswith('chroma')]

        for song_id, rating in collab_recs:
            content_recs = get_content_based_recommendations(song_id, song_features_df, top_n=num_recs)
            for _, row in content_recs.iterrows():
                # Ensure both song feature arrays are 2D for cosine_similarity
                song_features_row = song_features_df[song_features_df['filename'] == song_id][feature_columns].values
                row_features = row[feature_columns].values.reshape(1, -1)  # Ensure this is 2D (1 row, multiple columns)
                 # Clean song_features (single row)
                song_features_cleaned = np.array([
                [ast.literal_eval(str(x))[0] if isinstance(x, str) and x.startswith('[') else float(x) for x in row]
                for row in song_features_row
                ], dtype=float)

            # Clean feature_matrix (all songs)
                feature_matrix_cleaned = np.array([
                [ast.literal_eval(str(x))[0] if isinstance(x, str) and x.startswith('[') else float(x) for x in row]
                for row in row_features
                ], dtype=float)
            # Calculate content score based on cosine similarity
                content_score = cosine_similarity(song_features_cleaned, feature_matrix_cleaned).flatten()[0]
                hybrid_score = alpha * rating + (1 - alpha) * content_score  
                hybrid_recs.append((
                    row['filename'],
                    hybrid_score,
                    row.get('title', row['filename']),
                    row.get('artist', 'Unknown')
                ))

        hybrid_recs.sort(key=lambda x: x[1], reverse=True)
        return hybrid_recs[:num_recs]

    # Step 9: Get User Input and Show Recommendations
    user_input = st.text_input("Enter user ID to get hybrid recommendations")

    if user_input:
        hybrid_recs = hybrid_recommendations(user_input, top_n_collab, song_features_df)
        if hybrid_recs:
            st.subheader(f"🎧 Top 5 Hybrid Recommendations for User {user_input}")
            for song_id, score, title, artist in hybrid_recs:
                st.write(f"{title}")
        else:
            st.warning(f"No recommendations found for user '{user_input}'.")
else:
    st.error("CSV must have 'user', 'song', and 'rating' columns.")

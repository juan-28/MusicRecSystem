import streamlit as st
import base64
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import BooleanType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import nltk
from nltk.corpus import stopwords
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.sql.functions import col
import sys
# Other necessary imports
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.sql.functions import col

from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.sql.functions import col

def recommend_id(track_id, lsh_model, lyric_df, track_meta_df):
    # Filter the row with the given track_id
    target_lyric = lyric_df.filter(lyric_df['track_id'] == track_id).select('features')

    if target_lyric.count() > 0:
        target_features = target_lyric.first()['features']

        # Create a dataset with the target features
        target_dataset = spark.createDataFrame([(track_id, target_features)], ['track_id', 'features'])

        # Find approximate matches using LSH
        approx_matches = lsh_model.approxSimilarityJoin(target_dataset, lyric_df, 10, "distCol")
        
        # Exclude the exact match (target track itself)
        approx_matches = approx_matches.filter(~(col("datasetB.track_id") == track_id))
        

        # Get the top 10 recommendations
        top10_approx_matches = approx_matches.sort("distCol").limit(10)
        
        # Get the recommended track ids
        recommend_trackids = top10_approx_matches.select("datasetB.track_id")
        
        # Join with track_meta_df to get artist_name and title
        recommend_track = recommend_trackids.join(track_meta_df, recommend_trackids["track_id"] == track_meta_df["track_id"])
    else:
        recommend_track = spark.createDataFrame([], track_meta_df.schema)

    return recommend_track

def recommend_title(title, artist, lsh_model, lyric_df, track_meta_df, artist_sim_df):
    #recommended = []

    # Match the track id and artist id with the song title and artist name
    track_input = track_meta_df.filter((col('title') == title) & (col('artist_name') == artist)).select('track_id', 'artist_id')

    if not track_input.isEmpty():
        tid = track_input.first()['track_id']
        aid = track_input.first()['artist_id']
        
        # Get similar artists based on the artist_sim_df
        similar_artists = artist_sim_df.filter(artist_sim_df['target'] == aid).select('similar_artist').first()['similar_artist']
        
        # Include the input track in the recommendations
        #recommended.append(track_meta_df.filter(track_meta_df['track_id'] == tid).select('artist_name', 'title','track_id'))

        # Recommend based on approximate matching using LSH
        recommended = recommend_id(tid, lsh_model, lyric_df, track_meta_df).collect()
        
        # Recommend based on similar artists
        similar_artist_tracks = []
        for similar_artist_id in similar_artists:
            similar_artist_tracks.extend( track_meta_df.filter(col('artist_id') == similar_artist_id).select('track_id').rdd.flatMap(lambda x: x).collect())
        
        # Filter out the input track and tracks that were already recommended
        similar_artist_tracks = [track_id for track_id in similar_artist_tracks if track_id != tid and track_id not in [row['track_id'] for row in recommended]]

        # Recommend based on similar artists if available
        if similar_artist_tracks:
            recommended.extend(track_meta_df.filter(col('track_id').isin(similar_artist_tracks)).select('artist_name', 'title').take(10))

        titles = [row.title for row in recommended[:10]]
        artist_names = [row.artist_name for row in recommended[:10]]

    return pd.DataFrame({'Title': titles, 'Artist Name': artist_names})

# Initialize Spark session
spark = SparkSession.builder.appName("MusicRecommendation").getOrCreate()

#Preprocessing

artist_sim = spark.read.parquet("cleaned_artist_similarity.parquet")
# Load the tracks metadata
track_meta = spark.read.csv('./cleaned_data/tracks_metadata.csv', header=True, inferSchema=True)

# Assuming artist_sim is your PySpark DataFrame and 'new' is the column with string representation of lists
scaledData = spark.read.parquet("tf_idf_scaled.parquet")

tf_model_path = "./lsh_model_tfidf"
lsh_model_tfidf = BucketedRandomProjectionLSHModel.load(tf_model_path)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local('./background.png')    

title_input = st.text_input(":green[Title]")
artist_input = st.text_input(":green[Artist]")

if st.button("Recommend"):
    if title_input and artist_input:
        # Creating 4 columns for each recommendation type
        #col1, col2, col3, col4 = st.columns(4)

        tf_idf_recommendations = recommend_title(title_input, artist_input, lsh_model_tfidf, scaledData, track_meta, artist_sim)
        # Limit the DataFrame to the top 10 rows

        # Define a CSS style to change the text color to green
        green_text_style = '<style>tbody tr {color: green;}</style>'

        # Display the table with the green text color
        st.markdown(green_text_style, unsafe_allow_html=True)
        st.table(tf_idf_recommendations)

    else:
        st.error("Please enter both title and artist.")

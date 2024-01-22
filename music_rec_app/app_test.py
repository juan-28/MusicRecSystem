from pyspark.sql.functions import col, lit
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import os

spark = SparkSession.builder.appName("AppTest").getOrCreate()

#directory of the app_test.py
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct  paths relative to script_dir
meta_parquet_file_path = os.path.join(script_dir, "meta_parq")
int_parquet_file_path = os.path.join(script_dir, "int_parq")
model_path = os.path.join(script_dir, "model")

#Print the paths to verify
#print("Metadata Path:", meta_parquet_file_path)
#print("Interactions Path:", int_parquet_file_path)
#print("Model Path:", model_path)

metadata_df = spark.read.parquet(meta_parquet_file_path)
raw_plays_df_with_int_ids = spark.read.parquet(int_parquet_file_path)

if os.path.isdir(model_path):
    best_model = ALSModel.load(model_path)
else:
    print(f"Model path does not exist: {model_path}")
    
    
user_id_input = input("Enter User ID: ")
try:
    user_id = int(user_id_input)
except ValueError:
    print("Please enter a valid integer for User ID.")


def recommend_songs_for_user(user_id, best_model, raw_plays_df_with_int_ids, metadata_df):
    # Filter out the songs listened by the user
    listened_songs = raw_plays_df_with_int_ids.filter(raw_plays_df_with_int_ids.new_userId == user_id) \
                                 .join(metadata_df, 'songId') \
                                 .select('new_songId', 'artist_name', 'title')

    # Generate list of listened songs
    listened_songs_list = [song['new_songId'] for song in listened_songs.collect()]

    # Display songs the user has listened to
    print('Songs user has listened to:')
    listened_songs.select('artist_name', 'title').show()

    # Generate DataFrame of songs not listened by the user
    unlistened_songs = raw_plays_df_with_int_ids.filter(~ raw_plays_df_with_int_ids['new_songId'].isin(listened_songs_list)) \
                                   .select('new_songId').withColumn('new_userId', lit(user_id)).distinct()

    # Ensure correct data types
    unlistened_songs = unlistened_songs.withColumn("new_songId", col("new_songId").cast("integer"))
    unlistened_songs = unlistened_songs.withColumn("new_userId", col("new_userId").cast("integer"))

    # Feed unlistened songs into the model for prediction
    predicted_listens = best_model.transform(unlistened_songs)

    # Remove NaN values from predictions
    predicted_listens = predicted_listens.filter(predicted_listens['prediction'] != float('nan'))
    
    predicted_listens = predicted_listens.withColumn("new_songId", col("new_songId").cast("integer"))
    predicted_listens = predicted_listens.withColumn("new_userId", col("new_userId").cast("integer"))
    

# print output
    print('Predicted Songs:')
    predicted_listens.join(raw_plays_df_with_int_ids, 'new_songId') \
                 .join(metadata_df, 'songId') \
                 .select('artist_name', 'title', 'prediction') \
                 .distinct() \
                 .orderBy('prediction', ascending = False) \
                 .show(10)  

    return predicted_listens



recommended_songs_df = recommend_songs_for_user(user_id, best_model, raw_plays_df_with_int_ids, metadata_df)


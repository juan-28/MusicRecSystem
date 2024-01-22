import os
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id

def load_data(base_dir):
    # Define file paths
    triplets_filename = os.path.join(base_dir, 'train_triplets.txt')
    songs2tracks_filename = os.path.join(base_dir, 'taste_profile_song_to_tracks.txt')
    metadata_filename = os.path.join(base_dir, 'track_metadata.csv')

    # Define schema
    plays_df_schema = StructType([
        StructField('userId', StringType()),
        StructField('songId', StringType()),
        StructField('Plays', IntegerType())
    ])
    songs2tracks_df_schema = StructType([
        StructField('songId', StringType()),
        StructField('trackId', StringType())
    ])
    metadata_df_schema = StructType(
[StructField('trackId', StringType()),
   StructField('title', StringType()),
   StructField('songId', StringType()),
   StructField('release', StringType()),
   StructField('artist_id', StringType()),
   StructField('artist_mbid', StringType()),
   StructField('artist_name', StringType()),
   StructField('duration', DoubleType()),
   StructField('artist_familiarity', DoubleType()),
   StructField('artist_hotttness', DoubleType()),
   StructField('year', IntegerType()),
   StructField('track_7digitalid', IntegerType()),
   StructField('shs_perf', DoubleType()),
   StructField('shs_work', DoubleType())]
)

    # Create SparkSession
    spark = SparkSession.builder.appName("MusicRec").getOrCreate()

    # Load the data
    raw_plays_df = spark.read.format('com.databricks.spark.csv').options(header=False, delimiter='\t').schema(plays_df_schema).load(triplets_filename)
    songs2tracks_df = spark.read.format('com.databricks.spark.csv').options(header=False, delimiter='\t').schema(songs2tracks_df_schema).load(songs2tracks_filename)
    metadata_df = spark.read.format('com.databricks.spark.csv').options(header=True, delimiter=',').schema(metadata_df_schema).load(metadata_filename)

    return raw_plays_df, songs2tracks_df, metadata_df

def prepare_transformed_data(raw_plays_df):
    userId_change = raw_plays_df.select('userId').distinct().select('userId', monotonically_increasing_id().alias('new_userId'))
    songId_change = raw_plays_df.select('songId').distinct().select('songId', monotonically_increasing_id().alias('new_songId'))
    metadata_df = metadata_df.withColumn("index", monotonically_increasing_id())
    
    unique_users = userId_change.count()
    unique_songs = songId_change.count()
    
    raw_plays_df_with_int_ids = raw_plays_df.join(userId_change, 'userId').join(songId_change, 'songId')
    raw_plays_df_with_int_ids = raw_plays_df_with_int_ids.filter(raw_plays_df_with_int_ids.new_userId < unique_users / 2)

    return raw_plays_df_with_int_ids


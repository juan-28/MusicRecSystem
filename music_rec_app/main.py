import sys
import logging
from data_loading import load_data, prepare_transformed_data
from model_trainer import train_als_model
from recommender import recommend_songs_for_user
from pyspark.sql.functions import col

def setup_logging():
    logging.basicConfig(filename='app.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info("Logging is set up.")

def main(user_id):
    setup_logging()
    
    try:
        base_dir = 's3://music-rec-project/MSD/'
        raw_plays_df, songs2tracks_df, metadata_df = load_data(base_dir)
    
        raw_plays_df_with_int_ids = prepare_transformed_data(raw_plays_df)
        parquet_path = "s3://music-rec-project/parquet/raw_plays_df_with_int_ids/"
        
        parquet_path_meta = "s3://music-rec-project/parquet/metadata_df/"

        raw_plays_df_with_int_ids.write.parquet(parquet_path)
        metadata_df.write.parquet(parquet_path_meta)

        # logging.info("raw_plays_df_with_int_ids: %s", raw_plays_df_with_int_ids.show(5))
        # logging.info("songs2tracks_df: %s", songs2tracks_df.show(5))
        # logging.info("metadata_df: %s", metadata_df.show(5))

        # Split data into training, test, and validation sets
        (training_df, rest_df) = raw_plays_df_with_int_ids.randomSplit([0.6, 0.4], seed=123)
        (validation_df, test_df) = rest_df.randomSplit([0.5, 0.5], seed=123)
        
        training_df = training_df.withColumn("new_songId", col("new_songId").cast("integer"))
        training_df = training_df.withColumn("new_userId", col("new_userId").cast("integer"))
        validation_df = validation_df.withColumn("new_songId", col("new_songId").cast("integer"))
        validation_df = validation_df.withColumn("new_userId", col("new_userId").cast("integer"))
        test_df = test_df.withColumn("new_songId", col("new_songId").cast("integer"))
        test_df = test_df.withColumn("new_userId", col("new_userId").cast("integer"))
        

        seed = 42

        best_model = train_als_model(training_df, validation_df, seed)
        recommended_songs_df = recommend_songs_for_user(user_id, best_model, raw_plays_df_with_int_ids, metadata_df)

        logging.info("Training completed. Best model: %s", best_model)
        logging.info("Recommendations: %s", recommended_songs_df)

    except Exception as e:
        logging.error("An error occurred: %s", e, exc_info=True)

if __name__ == "__main__":
    user_id = int(sys.argv[1])
    main(user_id)

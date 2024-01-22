from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from elephas.spark_model import SparkModel
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
import sys

def create_track_id_mapping(df):
    # Use DataFrame API for better performance
    return df.select("track_id_index", "track_id").distinct().toPandas().set_index("track_id_index").to_dict()['track_id']

def recommend_items_for_user(user_id, model, df, track_id_mapping, n=5):
    # Use vectorization for predictions
    user_vector = df.filter(df.user_id == user_id).select("user_id_encoded").first()["user_id_encoded"]
    track_vectors = df.select("track_id_index", "track_id_encoded").distinct().toPandas()

    user_vectors = np.array([user_vector.toArray()] * len(track_vectors))
    track_vectors_array = np.array(track_vectors['track_id_encoded'].tolist())
    combined_vectors = np.concatenate((user_vectors, track_vectors_array), axis=1)

    predictions = model.predict(combined_vectors)
    top_indices = np.argsort(predictions[:, 0])[::-1][:n]
    top_tracks = [track_id_mapping[track_vectors.iloc[idx]['track_id_index']] for idx in top_indices]

    return top_tracks

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Autoencoder_Recommendation").getOrCreate()
    df = spark.read.csv(sys.argv[1], sep='\t', header=False, inferSchema=True)
    df = df.limit(1000)
    df = df.withColumnRenamed('_c0', 'user_id').withColumnRenamed('_c1', 'track_id').withColumnRenamed('_c2', 'frequency')
    

    # String indexer for user_id and track_id
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_index")
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_index")

    # One-hot encoding
    encoder_user = OneHotEncoder(inputCol="user_id_index", outputCol="user_id_encoded")
    encoder_track = OneHotEncoder(inputCol="track_id_index", outputCol="track_id_encoded")

    # Pipeline
    pipeline = Pipeline(stages=[indexer_user, indexer_track, encoder_user, encoder_track])
    df_transformed = pipeline.fit(df).transform(df)

    # Assembling features
    assembler = VectorAssembler(inputCols=["user_id_encoded", "track_id_encoded"], outputCol="features")
    df_final = assembler.transform(df_transformed)

    # Autoencoder Model
    input_dim = df_final.select("features").first()[0].size  # Total size of user_id_encoded + track_id_encoded
    input_layer = Input(shape=(input_dim,), name='input_layer')

    # Encoder
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)

    # Decoder
    decoded = Dense(128, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # Compile Model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    # Preparing RDD for Elephas
    rdd = df_final.select("features").rdd.map(lambda row: (row.features.toArray(), row.features.toArray()))

    # Elephas Model
    spark_model = SparkModel(model, frequency='epoch', mode='synchronous')
    spark_model.fit(rdd, epochs=5, batch_size=32, verbose=0, validation_split=0.1)

    # Create the track_id to track_id_index mapping
    track_id_mapping = create_track_id_mapping(df_transformed)

    # Example usage
    recommendations = recommend_items_for_user('b80344d063b5ccb3212f76538f3d9e43d87dca9e', model, df_transformed, track_id_mapping, n=5)
    print(recommendations)

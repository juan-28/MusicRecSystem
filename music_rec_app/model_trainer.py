from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F

def train_als_model(training_df, validation_df, seed):
    # Initialize ALS learner
    als = ALS()

    # Set parameters for ALS
    als.setMaxIter(5)\
       .setSeed(seed)\
       .setItemCol("new_songId")\
       .setRatingCol("Plays")\
       .setUserCol("new_userId")

    # Create RMSE evaluator
    reg_eval = RegressionEvaluator(predictionCol="prediction", labelCol="Plays", metricName="rmse")

    # Hyperparameter tuning setup
    tolerance = 0.03
    ranks = [4, 8, 12, 16]
    regParams = [0.15, 0.2, 0.25]
    errors = [[0] * len(ranks)] * len(regParams)
    models = [[0] * len(ranks)] * len(regParams)
    min_error = float('inf')
    best_params = [-1, -1]

    # Hyperparameter tuning loop
    for i, regParam in enumerate(regParams):
        for j, rank in enumerate(ranks):
            # Set ALS parameters
            als.setParams(rank=rank, regParam=regParam)

            # Train the model
            model = als.fit(training_df)

            # Make predictions
            predict_df = model.transform(validation_df)

            # Remove NaN values from predictions
            predicted_plays_df = predict_df.filter(predict_df.prediction != float('nan'))
            predicted_plays_df = predicted_plays_df.withColumn("prediction", F.abs(F.round(predicted_plays_df["prediction"], 0)))

            error = reg_eval.evaluate(predicted_plays_df)
            errors[i][j] = error
            models[i][j] = model
            print(f'For rank {rank}, regularization parameter {regParam} the RMSE is {error}')

            # Update best model 
            if error < min_error:
                min_error = error
                best_params = [i, j]

    # Set best model parameters
    als.setRegParam(regParams[best_params[0]])
    als.setRank(ranks[best_params[1]])
    print(f'The best model was trained with regularization parameter {regParams[best_params[0]]}')
    print(f'The best model was trained with rank {ranks[best_params[1]]}')
    
    models[best_params[0]][best_params[1]].write().overwrite().save(model_path)

    return models[best_params[0]][best_params[1]]

model_path = "s3://music-rec-project/model/" 




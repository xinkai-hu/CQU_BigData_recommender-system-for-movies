"""This file implemented a recommender system withPySpark built-in ALS algorithm.
If you read data from Neo4j database, do not forget to start it."""

import argparse
from argparse import Namespace

from pyspark import RDD
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import DataFrameReader


def get_config() -> Namespace:
    """Configuration when running this file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j-host", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-username", default="neo4j")
    parser.add_argument("--neo4j-passwd", default="20214919")
    parser.add_argument("--full-data", default="dataset/ml-100k/u.data")
    parser.add_argument("--train-data", default="dataset/ml-100k/u1.base")
    parser.add_argument("--test-data", default="dataset/ml-100k/u1.test")
    args = parser.parse_args()
    return args


def get_session() -> SparkSession:
    """Configurations for spark session."""
    spark: SparkSession = (
        SparkSession
        .builder
        .master("local")
        .appName("RecommenderSystem-ALS")
        .config("spark.jars", "jars/neo4j-connector-apache-spark-5.2.0/neo4j-connector-apache-spark_2.12-5.2.0_for_spark_3.jar")
        .getOrCreate()
    )
    return spark


def load_dataset_csv(
    data_path: str
) -> DataFrame:
    """Load dataset from `data_path`.

    Data in `data_path` should be organized as:
        `user_id: int`\t`movie_id: int`\t`rating: float`\t`timestamp: int`\n
    No headers in the table.
    """
    trainingset_rdd: RDD = (
        spark.read
        .text(data_path).rdd
        .map(
            lambda row: (
                row.value.split("\t")
            )
        )
        .map(
            lambda p: Row(
                user_id=int(p[0]),
                movie_id=int(p[1]),
                rating=float(p[2]),
                timestamp=int(p[3]),
            )
        )
    )

    return trainingset_rdd.toDF()


def read_query(
    query: str
) -> DataFrame:
    """Run a read-only Cypher query in connected Neo4j database."""
    db_reader: DataFrameReader = (
        spark.read
        .format("org.neo4j.spark.DataSource")
        .option("url", args.neo4j_host)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", args.neo4j_username)
        .option("authentication.basic.password", args.neo4j_passwd)
        .option("access.mode", "read")
    )

    return (
        db_reader
        .option("query", query)
        .load()
    )


def fit_model_ALS(
    trainingset_df: DataFrame
) -> tuple:
    """Fit ALS model with trainingset loaded by `load_dataset`."""
    print("Fitting model...")

    als = ALS(
        userCol="user_id", 
        itemCol="movie_id",
        ratingCol="rating", 
        nonnegative=True
    )

    model = (
        als
        .fit(trainingset_df)
        .setPredictionCol("score")
    )

    return als, model


def recommend(
    user_id: int,
    k: int,
    model: ALSModel
) -> DataFrame:
    """Recommend `k` movies with the highest score for `user_id`."""
    user = spark.createDataFrame([{"user_id": user_id}])

    return (
        model
        .recommendForUserSubset(user, k)
        .select("recommendations").rdd
        .flatMap(
            lambda row: row.recommendations
        )
        .toDF()
    )


if __name__ == "__main__":
    args = get_config()
    spark = get_session()

    """Loading the dataset."""
    """Un-comment the following lines to load splited dataset from csv files."""
    # train_df = load_dataset_csv(args.train_data)
    # test_df = load_dataset_csv(args.test_data)

    """Un-comment the following lines to load dataset from Neo4j database."""
    df = read_query(
        r"MATCH (u:User)-[r:Rate]->(m:Movie) "
        r"RETURN u.UserID AS user_id, "
        r"m.MovieID AS movie_id, "
        r"r.Rating AS rating"
    )
    train_df, test_df = df.randomSplit([0.8, 0.2])

    """Un-comment the following lines to load full dataset from csv files."""
    # df = load_dataset_csv(args.full_data)
    # train_df, test_df = df.randomSplit([0.8, 0.2])


    """Fitting and then evaluating the ALS model."""
    als, model = fit_model_ALS(train_df)

    pred_df = model.transform(test_df)
    pred_df = pred_df.na.drop()
    evaluator = RegressionEvaluator(
        predictionCol="score",
        labelCol="rating", 
        metricName="rmse"
    )

    print("Test RMSE:", evaluator.evaluate(pred_df))


    """Recommendation loops."""
    while True:
        user_id = int(input("Enter user ID: "))
        num = int(input("Enter recommendation number: "))
        rec_df = recommend(user_id, num, model)
        rec_df.show()

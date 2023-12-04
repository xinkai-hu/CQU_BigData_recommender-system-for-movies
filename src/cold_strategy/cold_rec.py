"""This file implemented the cold strategies for the recommender system.

For a new movie that is rated by users less than the required number, 
the strategy is seperate if from the majorities.

For a new user who rated less than `k` movies, the strategy is recommending 
the movies which have more than `m` ratings and have the highest average rating scores.

Before you run this file, please start Neo4j database."""

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import DataFrameReader


parser = argparse.ArgumentParser()
parser.add_argument("--neo4j-host", default="neo4j://localhost:7687")
parser.add_argument("--neo4j-username", default="neo4j")
parser.add_argument("--neo4j-passwd", default="20214919")
args = parser.parse_args()


def read_query(query) -> DataFrame:
    """Run a read-only Cypher query in connected Neo4j database."""
    spark: SparkSession = (
        SparkSession.builder
        .master("local")
        .appName("Connect Neo4j")
        .config("spark.jars", "jars/neo4j-connector-apache-spark-5.2.0/neo4j-connector-apache-spark_2.12-5.2.0_for_spark_3.jar")
        .getOrCreate()
    )
    
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



def rec_for_new_user(user_id: int, k: int, m: int) -> DataFrame:
    """Recommend `k` movies with more than `m` ratings 
    and with the highest mean rating. `user_id` is not used.
    """
    return read_query(
        r"MATCH (User)-[r:Rate]->(m:Movie) "
        r"WITH m, COUNT(r) AS cnt WHERE cnt > %d "
        r"MATCH (User)-[r:Rate]->(m) "
        r"RETURN m.MovieID AS movie_id, "
        r"AVG(r.Rating) AS avg_rating "
        r"ORDER BY avg_rating DESC"
        % m
    ).head(k)


def get_new_movie(k: int) -> DataFrame:
    """Get movies with less than `k` ratings as new movies."""
    return read_query(
        r"MATCH (:User)-[r:Rate]->(m:Movie) "
        r"WITH m, COUNT(r) AS cnt "
        r"WHERE cnt < %d "
        r"RETURN m.MovieID AS movie_id" 
        % k
    )

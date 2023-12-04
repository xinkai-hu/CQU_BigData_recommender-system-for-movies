"""This file implemented a simple real-time recommendater system with PySpark Structured Streaming.

Before start real-time recommendation, you should start Neo4j database
and run `nc -lk 23333` to simulate a data streaming with socket."""

import argparse
from argparse import Namespace
from pyspark.sql import DataFrame
from pyspark.sql import Row
from pyspark.sql import DataFrameReader
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
from pyspark.sql.functions import get
from pyspark.sql.functions import split
from neo4j import GraphDatabase
import numpy as np


def get_config() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j-host", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-username", default="neo4j")
    parser.add_argument("--neo4j-passwd", default="20214919")
    parser.add_argument("--num-users", type=int, default=943)
    parser.add_argument("--num-movies", type=int, default=1682)
    parser.add_argument("--features_dim", type=int, default=10)
    parser.add_argument("--model", default="saved_model/model-1m-topn.pth")
    args = parser.parse_args()
    return args


def get_session() -> SparkSession:
    spark: SparkSession = (
        SparkSession.builder
        .master("local")
        .appName("Streaming Recommendation")
        .config("spark.jars", "jars/neo4j-connector-apache-spark-5.2.0/neo4j-connector-apache-spark_2.12-5.2.0_for_spark_3.jar")
        .getOrCreate()
    )
    return spark


def write_query(query) -> DataFrame:
    """Write a query into Neo4j database."""
    driver = GraphDatabase().driver(args.neo4j_host, auth=(args.neo4j_username, args.neo4j_passwd))
    with driver.session() as session:
        session.execute_write(lambda tx, **msg: tx.run(query, **msg))
    driver.close()


def read_query(query) -> DataFrame:
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


def init_user_interest() -> DataFrame:
    """Save interest vector as a string."""
    write_query("MATCH (u:User) SET u.Interest = '{}';".format(",".join(str(v) for v in [0.0] * args.features_dim)))


def update_user_interest(batch_df: DataFrame, batch_id: int) -> None:
    """Notice: you can not write the batch_df results into any other DataFrames. 
    That is why we get every thing by Neo4j database.

    Core of streaming recommendation.

    When a new record <user_i, movie_j, r> comes, we read from database for interest vector U of user_i.
    Then update the interest vector with the tag vector T of movie_j.
    The formular is 
        U := lambda * r * U + (1 - lambda * r) * T
    where lambda is a custom constant (here we set lambda = 0.8).
    Cosine similarity is then calculated to evaluate user_i's preference to the movies.
    Movies with Top-N cosine similarity are recommended to user_i.
    """
    print("+-------- batch {} begin -----------+".format(batch_id))
    if batch_df.first() is not None:
        """Get tag vector of movie_j."""
        tags = batch_df.first().tags
        """Get interest vector."""
        data = read_query(r"MATCH (u:User{UserID:%d}) RETURN u.UserID AS user_id, u.Interest AS interest" % (batch_df.first().user_id))
        interest = [float(v) for v in data.first().interest.split(",")]
        """Update interest vector."""
        interest = np.array(interest) * (1 - 0.05 * batch_df.first().rating) + np.array(tags) * 0.05 * batch_df.first().rating
        """Write back to database."""
        write_query(
            r"MATCH (u:User{UserID:%d}) "
            r"SET u.Interest = '%s';" 
            % (batch_df.first().user_id, ",".join(str(v) for v in interest.tolist())))
        """Make recommendation."""
        rec = (
            movie_genres
            .rdd
            .map(lambda row: Row(
                movie_id=row.movie_id, 
                score=np.dot(row.tags, interest) / (np.linalg.norm(row.tags) * np.linalg.norm(interest) + 1e-8))
            )
            .top(10, key=lambda row: row.score)
        )
        print("interest:\n", interest)
        print("recommendation:\n", "\n".join(str(r) for r in rec))

        """Un-comment the following lines to write results into Neo4j database."""
        # write_query(r"MATCH (:User{UserID:%d})-[r:Recommend]->() DELETE r;" % batch_df.first().user_id)
        # for row in rec:
        #     write_query(
        #         r"MATCH (u:User{UserID:%d}) WITH u "
        #         r"MATCH (m:Movie{MovieID:%d}) WITH u, m "
        #         r"MERGE (u)-[:Recommend{Score:%f}]->(m);" 
        #         % (batch_df.first().user_id, row.movie_id, row.score)
        #     )
    print("+--------  batch {} end  -----------+\n".format(batch_id))


def get_multi_hot(genres: str) -> list:
    """Get the tag vector from genres. If the k-th tag is in genres of the movie, 
    then the k-th item of tag vector should be 1."""
    genres = set(genres.split("|"))
    return [1 if key in genres else 0 for key in GENERS]


def get_movie_genres():
    """Read movies from database and process their tag vectors."""
    data: DataFrame = read_query(
        r"MATCH (m:Movie) "
        r"RETURN m.MovieID AS movie_id, "
        r"m.Genres AS genres")
    data = (
        data.rdd
        .map(lambda row: Row(
            movie_id=row.movie_id,
            tags=get_multi_hot(row.genres)
        ))
    )
    return data.toDF()


def get_movie_features():
    """Read saved movie features."""
    data: DataFrame = read_query(
        r"MATCH (m:Movie) "
        r"RETURN m.MovieID AS movie_id, "
        r"m.Features AS features")
    data = (
        data.rdd
        .filter(lambda row: row.features is not None)
        .map(lambda row: Row(
            movie_id=row.movie_id,
            tags=[float(v) for v in row.features.split(",")]
        ))
    )
    return data.toDF()


def read_streaming():
    """parse string formatted in `user_id movie_id rating timestamp`."""
    (
        # Simulate a real-time data streaming by socket.
        spark
        .readStream
        .format("socket")
        .option("host", "localhost")
        .option("port", 23333)
        .load()
        # Parse received value string to a <user_id, movie_id, rating, timestamp> record.
        .withColumnRenamed("value", "record")
        .select(split(col("record"), " ").alias("record"))
        .select(
            get("record", 0).alias("user_id").cast(IntegerType()), 
            get("record", 1).alias("movie_id").cast(IntegerType()), 
            get("record", 2).alias("rating").cast(FloatType()),
            get("record", 3).alias("timestamp").cast(IntegerType())
        )
        .join(movie_genres, "movie_id", "left")
        .writeStream
        .outputMode(outputMode="append")
        # Output processed record into the function so that to save and update results.
        # foreachBatch can replate format to set customed output method.
        .foreachBatch(lambda batch_df, batch_id: update_user_interest(batch_df, batch_id))
        # Start reading and wait for real-time input streaming.
        .start()
        # Wait for inputs.
        .awaitTermination()
    )


if __name__ == "__main__":
    GENERS = [
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"
    ]
    args = get_config()
    spark = get_session()
    # movie_genres = get_movie_genres()
    movie_genres = get_movie_features()
    init_user_interest()
    users = read_query(
        r"MATCH (u:User) "
        r"RETURN u.UserID AS user_id, "
        r"u.interest AS interest")
    read_streaming()

"""This file implemented the recommendation logics with a pre-trained PyTorch model.
Make sure that the `num-users` and `num-movies` arguments fit the selected `model`.
Before you run this file, please start Neo4j database."""

import argparse
from argparse import Namespace

import torch
from neo4j import GraphDatabase
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import DataFrameReader
from torch import nn


def get_config() -> Namespace:
    parser = argparse.ArgumentParser()
    """Configurations for MovieLens 100k."""
    parser.add_argument("--num-users", type=int, default=943)
    parser.add_argument("--num-movies", type=int, default=1682)

    """Configurations for MovieLens 10M."""
    # parser.add_argument("--num-users", type=int, default=71567)
    # parser.add_argument("--num-movies", type=int, default=65133)

    parser.add_argument("--model", default="saved_model/model-100k-rmse.pth")
    parser.add_argument("--neo4j-host", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-username", default="neo4j")
    parser.add_argument("--neo4j-passwd", default="20214919")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return args


def get_session() -> SparkSession:
    """Configurations for spark session."""
    spark: SparkSession = (
        SparkSession.builder
        .master("local")
        .appName("Connect Neo4j")
        .config("spark.jars", "jars/neo4j-connector-apache-spark-5.2.0/neo4j-connector-apache-spark_2.12-5.2.0_for_spark_3.jar")
        .getOrCreate()
    )

    return spark


class MyModel(nn.Module):
    """Declaration of the model."""
    def __init__(self, base) -> None:
        super().__init__()
        self.base = base

    def forward(self, x):
        return self.base(x.T)


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


def write_query(query) -> None:
    """Write a query into Neo4j database."""
    driver = GraphDatabase().driver(args.neo4j_host, auth=(args.neo4j_username, args.neo4j_passwd))
    with driver.session() as session:
        session.execute_write(lambda tx, **msg: tx.run(query, **msg))
    driver.close()


def write_rec(user_id, rec) -> None:
    """Write recommendation results into Neo4j database."""
    for movie_id, score in rec:
        write_query(
            r"MATCH (u:User{UserID:%d}) WITH u "
            r"MATCH (m:Movie{MovieID:%d}) WITH u, m "
            r"MERGE (u)-[:Recommend{Score:%f}]->(m)"
            % (user_id, movie_id, score)
        )


if __name__ == "__main__":
    args = get_config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spark = get_session()

    """Loading the pre-trained model"""
    model = torch.load(args.model)

    """Loading movie list."""
    movie_index = list(range(args.num_users, args.num_users + args.num_movies))

    """Recommendation loops."""
    while True:
        user_id = int(input("Enter user ID: "))
        num = int(input("Enter recommendation number: "))

        """Movies that have been rated by the user will not be a candidate."""
        excepted = set(
            read_query(
                r"MATCH (User{UserID:%d})-[Rate]->(m:Movie) "
                r"RETURN m.MovieID" 
                % user_id
            )
            .dropDuplicates()
            .toPandas()
            .to_numpy()
            .reshape(-1)
            .tolist()
        )

        """Produce a tensor for the model and then predict the score."""
        x = torch.tensor([
            [user_id - 1] * len(movie_index),
            movie_index
        ]).to(device).T

        pred = model(x)
        pred = { index + 1 : score.item() for index, score in enumerate(pred) }
        pred = sorted(pred.items(), key=lambda kv: kv[1], reverse=True)
        pred = [ item for item in pred if item[0] not in excepted ][:num]
        print(pred)

        """Write the recommendation results into Neo4j database."""
        write_rec(user_id, pred)

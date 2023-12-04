"""This file tests the connection of Neo4j database."""

from neo4j import GraphDatabase
import argparse
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameReader

parser = argparse.ArgumentParser()
parser.add_argument("--neo4j-host", default="neo4j://localhost:7687")
parser.add_argument("--neo4j-username", default="neo4j")
parser.add_argument("--neo4j-passwd", default="20214919")
args = parser.parse_args()


def read_query(query) -> DataFrame:
    spark: SparkSession = (
        SparkSession.builder
        .master("local")
        .appName("Read Neo4j")
        .config("spark.jars", "jars/neo4j-connector-apache-spark-5.2.0/neo4j-connector-apache-spark_2.12-5.2.0_for_spark_3.jar")
        .getOrCreate())
    
    db_reader: DataFrameReader = (
        spark.read
        .format("org.neo4j.spark.DataSource")
        .option("url", args.neo4j_host)
        .option("authentication.type", "basic")
        .option("authentication.basic.username", args.neo4j_username)
        .option("authentication.basic.password", args.neo4j_passwd)
        .option("access.mode", "read"))

    return (
        db_reader
        .option("query", query)
        .load())


def write_query(query: str) -> DataFrame:
    driver = GraphDatabase().driver(args.neo4j_host, auth=(args.neo4j_username, args.neo4j_passwd))
    with driver.session() as session:
        session.execute_write(lambda tx, **msg: tx.run(query, **msg))
    driver.close()


result = read_query(
    r"MATCH (u:User)-[r:Rate]->(m:Movie) "
    r"RETURN u.UserID AS user_id, "
    r"m.MovieID AS movie_id, "
    r"r.Rating AS Rating, "
    r"r.Timestamp AS timestamp")
print("Writing...")

# write_query(r"MATCH (u:User{UserID:1}) WITH u MATCH (m:Movie{MovieID:1}) WITH u, m MERGE (u)-[:Recommend{Score:1.2, Timestamp:120}]->(m);")

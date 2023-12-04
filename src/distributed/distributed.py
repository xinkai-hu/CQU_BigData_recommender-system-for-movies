"""Distributed training with `horovod.spark.torch`.

Run command:
```
$ python3 src/distributed/train.py --master=localhost [other args...]
```

Reference to `https://horovod.readthedocs.io/en/stable/spark_include.html#spark-cluster-setup`
for clustering training setup.
"""

import argparse
import sys

from neo4j import GraphDatabase
import horovod.spark.torch as hvd
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store
from pyspark import SparkConf
from pyspark import RDD
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import DenseVector
from pyspark.sql import DataFrame
from pyspark.sql import DataFrameReader
from pyspark.sql import Row
from pyspark.sql import SparkSession
import torch
from torch import nn
from torch import optim
from torch_geometric.nn.models import LightGCN


def get_config():
    """Configuration for running this file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default="local")
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--num-users", type=int, default=943)
    parser.add_argument("--num-movies", type=int, default=1682)
    parser.add_argument("--neo4j-host", default="neo4j://localhost:7687")
    parser.add_argument("--neo4j-username", default="neo4j")
    parser.add_argument("--neo4j-passwd", default="20214919")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--train-data", default="dataset/ml-100k/u1.base")
    parser.add_argument("--test-data", default="dataset/ml-100k/u1.test")
    parser.add_argument("--full-data", default="dataset/ml-100k/u.data")
    parser.add_argument("--store-dir", default="tmp")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="src/distributed/model.pth")
    parser.add_argument("--backward-passes-per-step", type=int, default=1)
    args = parser.parse_args()
    return args


def get_session() -> SparkSession:
    conf = (
        SparkConf()
        .setAppName("Rec Train")
        .set("spark.sql.shuffle.partitions", "16")
    )
    if args.master:
        conf.setMaster(args.master)
    elif args.num_proc:
        conf.setMaster("local[{}]".format(args.num_proc))
    spark: SparkSession = (
        SparkSession.builder
        .config(conf=conf)
        .config("spark.jars", "jars/neo4j-connector-apache-spark-5.2.0/neo4j-connector-apache-spark_2.12-5.2.0_for_spark_3.jar")
        .getOrCreate()
    )
    
    return spark


class MyModel(nn.Module):
    """Adaptor that makes PyG model works like normal `torch.nn.model`."""
    def __init__(self, base) -> None:
        super().__init__()
        self.base = base

    def forward(self, x):
        return self.base(x.long().T)


def load_dataset_csv(
    data_path: str
) -> DataFrame:
    """Load dataset from `data_path`.

    Data in `data_path` should be organized as:
        `user_id: int`\t`movie_id: int`\t`rating: float`\t`timestamp: int`\n
    No headers in the table.
    """
    dataset_rdd: RDD = (
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

    return dataset_rdd.toDF()


def read_query(query) -> DataFrame:
    """Run a read-only Cypher query in connected Neo4j database."""
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
        .load()
    )


def write_query(query: str) -> DataFrame:
    """Write a query into Neo4j database."""
    driver = GraphDatabase().driver(args.neo4j_host, auth=(args.neo4j_username, args.neo4j_passwd))
    with driver.session() as session:
        session.execute_write(lambda tx, **msg: tx.run(query, **msg))
    driver.close()


if __name__ == "__main__":
    args = get_config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spark = get_session()
    store = Store.create(args.store_dir)

    """Un-comment the following lines to load dataset from Neo4j database."""
    df = read_query(
        r"MATCH (u:User)-[r:Rate]->(m:Movie) "
        r"RETURN u.UserID AS user_id, "
        r"m.MovieID AS movie_id, "
        r"r.Rating AS rating"
    )

    """Un-comment the following lines to load full dataset from csv file."""
    # df = load_dataset_csv(args.full_data)

    rdd = (
        df.rdd
        .map(
            lambda row: Row(
                features=DenseVector([row.user_id - 1, row.movie_id - 1 + args.num_users]), 
                label=row.rating
            )
        )
    )
    df = spark.createDataFrame(rdd, ["features", "label"])
    train_df, test_df = df.randomSplit([0.8, 0.2])

    num_nodes = args.num_users + args.num_movies
    model = MyModel(LightGCN(num_nodes, args.embedding_dim, args.num_layers))
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    backend = SparkBackend(
        num_proc=args.num_proc,
        stdout=sys.stdout, 
        stderr=sys.stderr,
        prefix_output_with_timestamp=True
    )
    torch_estimator = hvd.TorchEstimator(
        backend=backend,
        store=store,
        model=model,
        optimizer=optimizer,
        loss=lambda input, target: loss_fn(input, target),
        input_shapes=[[-1, 2]],
        feature_cols=["features"],
        label_cols=["label"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation=0.1,
        backward_passes_per_step=args.backward_passes_per_step,
        verbose=1
    )

    """Fitting the model."""
    torch_model: hvd.TorchModel = (
        torch_estimator
        .fit(train_df)
        .setOutputCols(["score"])
    )

    torch.save(torch_model.getModel(), args.model)

    evaluator = RegressionEvaluator(
        predictionCol="score", 
        labelCol="label", 
        metricName="rmse"
    )

    """Un-commend the following lines to enable RMSE evaluation. It takes time."""
    # print("Evaluating model...")
    # pred_df = torch_model.transform(test_df)
    # print("Test RMSE:", evaluator.evaluate(pred_df))

    movie_index_rdd = (
        read_query(
            r"MATCH (m:Movie) RETURN m.MovieID AS movie_id"
        )
        .rdd
        .map(
            lambda row: Row(
                movie_index=row.movie_id - 1 + args.num_users
            )
        )
    )

    """Recommendation loops."""
    while True:
        try:
            user_id = int(input("Enter user ID: "))
            num = int(input("Enter recommendation number: "))
        except:
            spark.stop()
            exit()

        """Movies that are rated by the user will not be in candidates."""
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

        features_rdd = movie_index_rdd.map(
            lambda row: Row(
                features=DenseVector([user_id - 1, row.movie_index])
            )
        )
        features_df = features_rdd.toDF()
        rec_df: DataFrame = torch_model.transform(features_df)

        rec_rdd = (
            rec_df.rdd
            .map(
                lambda row: Row(
                    movie_id=int(row.features[1]) + 1 - args.num_users, 
                    score=row.score
                )
            )
            .filter(
                lambda row: row.movie_id not in excepted
            )
        )

        rec_df = spark.createDataFrame(rec_rdd, ["movie_id", "score"])
        rec_df = rec_df.orderBy(rec_df.score.desc()).limit(num)

        """Write results into Neo4j database."""
        rec_df.foreach(
            lambda row: write_query(
                r"MATCH (u:User{UserID:%d}) WITH u "
                r"MATCH (m:Movie{MovieID:%d}) WITH u, m "
                r"MERGE (u)-[:Recommend{Score:%f}]->(m)"
                % (user_id, row.movie_id, row.score)
            )
        )

        print("Recommendations are inserted into Neo4j database.")

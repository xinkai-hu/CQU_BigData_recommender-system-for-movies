"""This file tests the environment for `horovod.spark.torch`.\n
It is also the expansion of EX3 Chinese MNIST recognition task.

Run command:
$ python3 src/test/horovod_test.py --master=localhost
"""

import argparse
import os
import sys

import horovod.spark.torch as hvd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store
from pyspark import SparkConf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import SparseVector
from pyspark.rdd import RDD
from pyspark.sql.functions import udf
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import types as T


parser = argparse.ArgumentParser(
    description="Chinese Handwriting recognition",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--master",
    help="spark master to connect to")
parser.add_argument(
    "--num-proc", type=int,
    help="number of worker processes for training, default: `spark.default.parallelism`")
parser.add_argument(
    "--batch-size", type=int, default=128,
    help="input batch size for training")
parser.add_argument(
    "--epochs", type=int, default=20,
    help="number of epochs to train")
parser.add_argument(
    "--data-raw", default="/home/hadoop/big_data/data/Chinese_MNIST/RawDataset",
    help="path to raw dataset")
parser.add_argument(
    "--data-processed", default="/home/hadoop/big_data/data/Chinese_MNIST/processed",
    help="path to processed dataset")
parser.add_argument(
    "--work-dir", default="/home/hadoop/big_data/tmp",
    help="number of backward passes to perform before calling hvd.allreduce")
parser.add_argument(
    "--backward-passes-per-step", type=int, default=1,
    help="number of backward passes to perform before calling hvd.allreduce")


if __name__ == "__main__":
    args = parser.parse_args()

    conf = SparkConf().setAppName("pytorch_spark_mnist").set("spark.sql.shuffle.partitions", "16")
    if args.master:
        conf.setMaster(args.master)
    elif args.num_proc:
        conf.setMaster("local[{}]".format(args.num_proc))
    spark: SparkSession = SparkSession.builder.config(conf=conf).getOrCreate()

    # Setup our store for intermediate data
    store = Store.create(args.work_dir)

    if os.path.exists(args.data_processed):
        df = spark.read.parquet("file://" + args.data_processed)
    else:
        image: RDD = (
            spark.read.format("image")
            .option("dropInvalid", True)
            .load("file://" + args.data_raw)
            .rdd)

        image = (
            image
            .map(lambda row: row.image)
            .map(lambda row: Row(
                file=row.origin,
                data=row.data))
            .map(lambda row: Row(
                label=float(row.file.split("%7D")[0].split(",")[-1]),
                data=np.frombuffer(row.data, dtype=np.ubyte)))
            .map(lambda row: Row(label=row.label - 1, features=SparseVector(
                4096, { index : value for index, value in enumerate(row.data) if value != 0 }))))

        df = spark.createDataFrame(image, ["label", "features"])
        df.coalesce(16).write.mode("overwrite").parquet("file://" + args.data_processed)


    # Train/test split
    train_df, test_df = df.randomSplit([0.9, 0.1])

    # Define the PyTorch model without any Horovod-specific parameters
    class Net(nn.Module):
        """Simple CNN model."""
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(3380, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 15)

        def forward(self, features):
            x = features.float()
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 3380)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, training=self.training)
            return self.fc3(x)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss = nn.CrossEntropyLoss()

    # Train a Horovod Spark Estimator on the DataFrame
    backend = SparkBackend(
        num_proc=args.num_proc,
        stdout=sys.stdout, 
        stderr=sys.stderr,
        prefix_output_with_timestamp=True)
    torch_estimator = hvd.TorchEstimator(
        backend=backend,
        store=store,
        model=model,
        optimizer=optimizer,
        loss=lambda input, target: loss(input, target.long()),
        input_shapes=[[-1, 1, 64, 64]],
        feature_cols=["features"],
        label_cols=["label"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation=0.1,
        backward_passes_per_step=args.backward_passes_per_step,
        verbose=1)

    torch_model = torch_estimator.fit(train_df).setOutputCols(["label_prob"])
    torch.save(torch_model, "/home/hadoop/big_data/model.pth")
    # Evaluate the model on the held-out test DataFrame
    pred_df = torch_model.transform(test_df)

    argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
    pred_df = pred_df.withColumn("label_pred", argmax(pred_df.label_prob))
    evaluator = MulticlassClassificationEvaluator(
        predictionCol="label_pred", labelCol="label", metricName="accuracy")
    print("Test accuracy:", evaluator.evaluate(pred_df))

    spark.stop()

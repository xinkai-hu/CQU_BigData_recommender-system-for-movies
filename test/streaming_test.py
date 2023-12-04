"""This file tests streaming processing.
Use `nc -lk <port>` to simulize a real-time data streaming.

Three simple examples of streaming are provided."""

from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
from pyspark.sql.functions import desc
from pyspark.sql.functions import explode
from pyspark.sql.functions import get
from pyspark.sql.functions import shuffle
from pyspark.sql.functions import split


spark: SparkSession = (
    SparkSession.builder
    .master("local")
    .appName("Streaming test")
    .getOrCreate()
)

streaming_data = (
    spark.readStream
    .format("socket")
    .option("host", "localhost")
    .option("port", 23333)
    .load()
)

def word_count():
    """word count
    
    Example:
        <shell 1> $ nc -lk 23333
        <shell 2> $ python3 streaming_test.py
        <shell 1> $ hadoop spark spark cpp python java java javahadoop

        Result in <shell 1>:
        -------------------------------------------                                     
        Batch: 1
        -------------------------------------------
        +----------+-----+
        |      word|count|
        +----------+-----+
        |     spark|    2|
        |      java|    2|
        |javahadoop|    1|
        |    hadoop|    1|
        |       cpp|    1|
        |    python|    1|
        +----------+-----+
    """
    (    
        streaming_data
        .withColumnRenamed("value", "word")
        .withColumn("word", explode(split("word", " ")))
        .groupBy("word")
        .count()
        .orderBy(desc("count"))
        .writeStream
        .outputMode(outputMode="complete")
        .format("console")
        .start()
        .awaitTermination()
    )

def shuffle_sentence():
    """shuffle an input sentence

    Example:
        <shell 1> $ nc -lk 23333
        <shell 2> $ python3 streaming_test.py
        <shell 1> $ During handling of the above exception, another exception occurred:

        Result in <shell 1>:
        -------------------------------------------
        Batch: 1
        -------------------------------------------
        +----------+
        |       col|
        +----------+
        |  handling|
        |exception,|
        |       the|
        |        of|
        | exception|
        |     above|
        | occurred:|
        |   another|
        |    During|
        +----------+
    """
    (
        streaming_data
        .withColumnRenamed("value", "word")
        .select(explode(shuffle(split(col("word"), " "))))
        .writeStream
        .outputMode(outputMode="append")
        .format("console")
        .start()
        .awaitTermination()
    )


def get_user_movie_rating_timestamp():
    """parse string formatted in `user_id movie_id rating timestamp`.
    
    Example:
        <shell 1> $ nc -lk 23333
        <shell 2> $ python3 streaming_test.py
        <shell 1> $ 1 2 5 1222222222

        Result in <shell 1>:
        -------------------------------------------
        Batch: 1
        -------------------------------------------
        +-------+--------+------+----------+
        |user_id|movie_id|rating| timestamp|
        +-------+--------+------+----------+
        |      1|       2|   5.0|1222222222|
        +-------+--------+------+----------+
    """
    (
        streaming_data
        .withColumnRenamed("value", "record")
        .select(split(col("record"), " ").alias("record"))
        .select(
            get("record", 0).alias("user_id").cast(IntegerType()), 
            get("record", 1).alias("movie_id").cast(IntegerType()), 
            get("record", 2).alias("rating").cast(FloatType()),
            get("record", 3).alias("timestamp"))
        .writeStream
        .outputMode(outputMode="append")
        .format("console")
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    word_count()
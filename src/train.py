import configparser
from pathlib import Path
from src.logger import Logger
from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler

root_dir = Path(__file__).parent.parent
CONFIG_PATH = str(root_dir / 'config.ini')


class Trainer:
    def __init__(self):
        self.logger = Logger(show=True).get_logger(__name__)
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(CONFIG_PATH)

        self.config = config
        spark_conf = SparkConf().setAll(config['SPARK'].items())
        self.spark = SparkSession.builder \
            .appName("KMeans") \
            .master("local[*]") \
            .config(conf=spark_conf) \
            .getOrCreate()

        self.data_path = str(root_dir / config['DATA']['processed'])
        self.model_path = str(root_dir / config['MODEL']['model_path'])

    def train_pipeline(self, k=5):
        self.logger.info("Starting training...")

        df = self.spark.read.option("header", True) \
               .option("sep", "\t") \
               .option("inferSchema", True) \
               .csv(self.data_path)

        for c in df.columns:
            if dict(df.dtypes)[c] == "string":
                df = df.withColumn(c, col(c).cast(DoubleType()))

        numeric_cols = [c for c, t in df.dtypes if t in ("double", "int", "float", "bigint")]
        df = df.select(numeric_cols).na.fill(0.0)

        assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
        kmeans = KMeans(k=k, seed=42, featuresCol="scaled_features", predictionCol="cluster")

        pipeline = Pipeline(stages=[assembler, scaler, kmeans])
        pipeline_model = pipeline.fit(df)

        pipeline_model.write().overwrite().save(self.model_path)
        self.logger.info("Model successfully saved!")

    def stop(self):
        self.spark.stop()
        self.logger.info("SparkSession stopped")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_pipeline(k=5)
    trainer.stop()

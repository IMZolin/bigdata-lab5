import configparser
from pathlib import Path
from src.logger import Logger
import pyspark
from pyspark import SparkConf
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

root_dir = Path(__file__).parent.parent
CONFIG_PATH = str(root_dir / 'config.ini')
DATA_PATH = str(root_dir / 'data' / 'processed_products.csv')
MODEL_PATH = str(root_dir / 'model')


class Predictor:
    def __init__(self):
        self.logger = Logger(show=True).get_logger(__name__)
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(CONFIG_PATH)
        spark_conf = SparkConf().setAll(config['SPARK'].items())

        self.spark = SparkSession.builder \
            .appName("KMeans") \
            .master("local[*]") \
            .config(conf=spark_conf) \
            .getOrCreate()

        # Load saved pipeline model
        self.pipeline = PipelineModel.load(MODEL_PATH)
        self.logger.info("Model successfully loaded")

    def predict(self, df: pyspark.sql.DataFrame):
        """Return DataFrame with predicted cluster labels."""
        for c in df.columns:
            df = df.withColumn(c, col(c).cast("double"))
        df = df.na.fill(0)
        result_df = self.pipeline.transform(df)
        cols = df.columns + ["cluster"]  # Keep only original columns plus cluster label
        return result_df.select(*cols)

    def stop(self):
        self.spark.stop()
        self.logger.info("SparkSession stopped")


if __name__ == "__main__":
    pred = Predictor()
    df_new = pred.spark.read.option("header", True) \
                       .option("sep", "\t") \
                       .option("inferSchema", True) \
                       .csv(DATA_PATH)

    # Make predictions and display first 10 cluster labels
    labels = pred.predict(df_new).select('cluster')
    labels.show(10, truncate=False)
    pred.stop()

import configparser
from pathlib import Path
from logger import Logger
from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler


root_dir = Path(__file__).parent.parent
CONFIG_PATH = str(root_dir / 'config.ini')
DATA_PATH   = str(root_dir / 'data' / 'processed_products.csv')
MODEL_PATH  = str(root_dir / 'model')


class Trainer:
    def __init__(self):
        self.logger = Logger().get_logger(__name__)
import os
import shutil
import configparser
from pathlib import Path
from src.logger import Logger
from functools import reduce
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when

# Define important file paths relative to project root
root_dir = Path(__file__).parent.parent
CONFIG_PATH = str(root_dir / 'config.ini')
DATA_PATH = str(root_dir / 'data' / 'products.csv')
OUTPUT_PATH = str(root_dir / 'data' / 'processed_products.csv')


class DataMaker:
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
        
    def prepare_data(self):
        self.logger.info("Starting preprocessing...")
        
        # Load tab-separated product data with header and inferred schema
        df = self.spark.read.option("header", True) \
               .option("sep", "\t") \
               .option("inferSchema", True) \
               .csv(DATA_PATH)
        
        # Select nutrient-related columns (starting from index 88)
        nutrient_columns = df.columns[88:]
        df_nutrients = df.select(nutrient_columns)

        # Filter out rows where *all* selected columns are NULL
        df_cleaned = df_nutrients.filter(
            ~reduce(lambda a, b: a & b, [col(c).isNull() for c in df_nutrients.columns])
        )
        df_filled = df_cleaned.fillna(0.0)
        lower_bound = 0.0
        upper_bound = 1000.0

        # Compute median values for all columns
        median_exprs = [expr(f"percentile_approx(`{c}`, 0.5)").alias(c) for c in df_filled.columns]
        medians = df_filled.agg(*median_exprs).collect()[0].asDict()

        df_cleansed = df_filled
        # Replace values outside the valid range with the median
        for c in df_filled.columns:
            if c in medians and medians[c] is not None:
                median = medians[c]
                df_cleansed = df_cleansed.withColumn(
                    c,
                    when((col(c) < lower_bound) | (col(c) > upper_bound), median).otherwise(col(c))
                )
                temp_output_path = OUTPUT_PATH[:-4]

        # Save the cleaned data as a single TSV file
        df_cleansed.coalesce(1).write \
            .mode("overwrite") \
            .option("header", True) \
            .option("sep", "\t") \
            .csv(temp_output_path)

        # Rename Spark’s part file to the desired output file
        for file in os.listdir(temp_output_path):
            if file.startswith("part-00000"):
                shutil.move(os.path.join(temp_output_path, file), OUTPUT_PATH)
                break
        shutil.rmtree(temp_output_path)
        self.logger.info("Data successfully processed and saved!")

        
    def stop(self):
        self.spark.stop()
        self.logger.info("SparkSession stopped")


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.prepare_data()
    data_maker.stop()

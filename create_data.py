import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType
import os

def create_sample_data_spark_parquet(file_path):
    """Create sample Parquet data using PySpark"""

    # Start SparkSession
    spark = SparkSession.builder \
        .appName("SampleDataGenerator") \
        .getOrCreate()

    num_samples = 10

    # Generate fake image data (128x128 grayscale, flattened)
    images = np.random.rand(num_samples, 128 * 128).astype(np.float32)
    labels_2013 = np.random.rand(num_samples).astype(np.float32)
    labels_2014 = np.random.rand(num_samples).astype(np.float32)
    areas = [f"Area_{i}" for i in range(num_samples)]

    # Create data rows as a list of dicts
    data = [{
        'image_array': image.tolist(),
        'pct_2013': float(p13),
        'Area': area,
        'pct_2014': float(p14)
    } for image, p13, p14, area in zip(images, labels_2013, labels_2014, areas)]

    # Define schema
    schema = StructType([
        StructField("image_array", ArrayType(FloatType()), True),
        StructField("pct_2013", FloatType(), True),
        StructField("Area", StringType(), True),
        StructField("pct_2014", FloatType(), True)
    ])

    # Create Spark DataFrame
    df = spark.createDataFrame(data, schema=schema)

    # Write to Parquet
    df.write.mode("overwrite").parquet(file_path)
    print(f"Sample data written to {file_path}")
    spark.stop()

# Example usage:
create_sample_data_spark_parquet("sample_data.parquet")

# move using this
#  hdfs dfs -mv /user/prashu/sample_data.parquet /user/prashu/processed/sample_data.parquet
from pyspark.sql import SparkSession
import numpy as np

def generate_sample_dataset(file_path, num_rows=10):
    """
    Generate a sample Spark DataFrame with a dummy image input and label,
    then write it as a parquet file.
    """
    spark = SparkSession.builder.appName("GenerateSampleDataset").getOrCreate()

    # Build dummy data: each row has one "image" column and one label column "pct_2013".
    # Here, "image" contains a 128x128x1 array represented as nested lists.
    data = []
    for _ in range(num_rows):
        # Create a random image with the desired dimensions
        image = np.random.rand(128, 128, 1).tolist()  # convert numpy array to lists for Spark compatibility
        pct_2013 = float(np.random.rand())  # a random float label
        data.append((image, pct_2013))

    # Create a DataFrame with schema ["image", "pct_2013"]
    df = spark.createDataFrame(data, schema=["image", "pct_2013"])
    df.write.mode("overwrite").parquet(f"processed/{file_path}")
    print(f"Sample dataset generated and written to: processed/{file_path}")

if __name__ == "__main__":
    # Modify file path as needed (for local testing, you might use a local folder)
    sample_file = "test_data.parquet"
    generate_sample_dataset(sample_file, num_rows=10)


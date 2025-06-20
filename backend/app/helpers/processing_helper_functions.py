from pyspark.ml.feature import (
    Imputer,
    MinMaxScaler,
    Normalizer,
    StandardScaler,
    VectorAssembler,
    OneHotEncoder,
    StringIndexer,
)
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    FloatType,
    DecimalType,
    StringType,
    BooleanType,
)
from pyspark.sql.functions import col, udf, lit
from pyspark.sql import functions as F
from functools import reduce
import math
import time
from uuid import uuid4


"""
    I have tried to keep the functions optimal for large datasets, such that they can be run on a cluster with
    multiple executors effectively. if there is a need to change please ensure the same thing for the new code.
"""


def get_temp_col(base: str) -> str:
    """Generates unique temp column names using UUID"""
    return f"{base}_{uuid4().hex[:8]}"


def remove_outlier_by_IQR(dataframe, columns, factor=1.5):
    """
    Detects and treats outliers using IQR for multiple variables in a PySpark DataFrame,
    Removes the whole row if any column has an outlier.

    :param dataframe: The input PySpark DataFrame
    :param columns: A list of columns to apply IQR outlier treatment
    :param factor: The IQR factor to use for detecting outliers (default is 1.5)
    :return: The processed DataFrame with outliers treated
    """
    conditions = []
    for column in columns:
        # Calculate Q1, Q3, and IQR
        quantiles = dataframe.approxQuantile(column, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        # Define the upper and lower bounds for outliers
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        conditions.append(F.col(column).between(lower_bound, upper_bound))

    return dataframe.where(reduce(lambda a, b: a & b, conditions))


def normalize_column(df, column_name, method):
    """
    i) Normalizes a column in a PySpark DataFrame using specified normalization method
    Supported methods: 'min-max', 'z-score', 'l1', 'l2', 'linf'
    ii) It removes entire rows if any of the specified columns contains an outlier.
    iii) I'm calculating stats multiple times as required(it seems this will cause multiple scans),
        but a user will normalize a column only once (by any given method), so this is better,
        same reason for not computing stats for every column at once
    """

    if method == "Min-Max":
        stats = df.agg(
            F.min(F.col(column_name)).alias("min"),
            F.max(F.col(column_name)).alias("max"),
        ).first()
        min_val = stats["min"]
        max_val = stats["max"]

        # Handle constant column
        if (max_val - min_val) == 0:
            return df.withColumn(column_name, F.lit(0.0))
        return df.withColumn(
            column_name, (F.col(column_name) - min_val) / (max_val - min_val)
        )

    elif method == "Z-score":
        stats = df.agg(
            F.mean(F.col(column_name)).alias("mean"),
            F.stddev(F.col(column_name)).alias("stddev"),
        ).first()
        mean_val = stats["mean"]
        stddev_val = stats["stddev"] or 0  # Handle null for constant column

        if stddev_val == 0:
            return df.withColumn(column_name, F.lit(0.0))
        return df.withColumn(column_name, (F.col(column_name) - mean_val) / stddev_val)

    elif method == "L1 Norm":
        abs_sum = df.agg(F.sum(F.abs(F.col(column_name)))).first()[0]
        if abs_sum == 0:
            return df.withColumn(column_name, F.lit(0.0))
        return df.withColumn(column_name, F.col(column_name) / abs_sum)

    elif method == "L2 Norm":
        squared_sum = df.agg(F.sum(F.pow(F.col(column_name), 2))).first()[0]
        if squared_sum == 0:
            return df.withColumn(column_name, F.lit(0.0))
        l2_norm = math.sqrt(squared_sum)
        return df.withColumn(column_name, F.col(column_name) / l2_norm)

    elif method == "L inf Norm":
        abs_max = df.agg(F.max(F.abs(F.col(column_name)))).first()[0]
        if abs_max == 0:
            return df.withColumn(column_name, F.lit(0.0))
        return df.withColumn(column_name, F.col(column_name) / abs_max)

    else:
        print(
            f"Unsupported normalization method: {method} for the column {column_name}"
        )


def All_Column_Operations(df, step, numericCols, allCols):
    if step["operation"] == "Drop Null":
        return df.dropna(subset=allCols)

    elif step["operation"] == "Fill 0 Unknown False":
        return (
            df.fillna(0, subset=allCols)
            .fillna("unknown", subset=allCols)
            .fillna(False, subset=allCols)
        )

    elif step["operation"] == "Fill Mean":
        imputer = Imputer(
            inputCols=numericCols, outputCols=numericCols, strategy="mean"
        )
        return imputer.fit(df).transform(df)

    elif step["operation"] == "Fill Median":
        imputer = Imputer(
            inputCols=numericCols, outputCols=numericCols, strategy="median"
        )
        return imputer.fit(df).transform(df)

    elif step["operation"] == "Drop Duplicates":
        return df.dropDuplicates()

    elif step["operation"] in [
        "L1 Norm",
        "L2 Norm",
        "L inf Norm",
        "Min-Max",
        "Z-score",
    ]:
        tempcol_1 = get_temp_col("features")
        assembler = VectorAssembler(inputCols=numericCols, outputCol=tempcol_1)
        df = assembler.transform(df)

        tempcol_2 = get_temp_col("scaledfeatures")
        if step["operation"] == "L1 Norm":
            normalizer = Normalizer(inputCol=tempcol_1, outputCol=tempcol_2, p=1.0)
        elif step["operation"] == "L2 Norm":
            normalizer = Normalizer(inputCol=tempcol_1, outputCol=tempcol_2, p=2.0)
        elif step["operation"] == "L inf Norm":
            normalizer = Normalizer(
                inputCol=tempcol_1, outputCol=tempcol_2, p=float("inf")
            )
        elif step["operation"] == "Min-Max":
            scaler = MinMaxScaler(inputCol=tempcol_1, outputCol=tempcol_2)
        elif step["operation"] == "Z-score":
            scaler = StandardScaler(
                inputCol=tempcol_1, outputCol=tempcol_2, withStd=True, withMean=False
            )

        if step["operation"] in ["L1 Norm", "L2 Norm", "L inf Norm"]:
            df = normalizer.transform(df)
        else:
            df = scaler.fit(df).transform(df)

        col_to_idx = {col: idx for idx, col in enumerate(numericCols)}
        # this is optimized query for large datasets and needs a one time scan only
        ordered_cols = [
            (
                vector_to_array(F.col(tempcol_2))[col_to_idx[col]].alias(col)
                if col in numericCols
                else F.col(col)
            )
            for col in df.columns
            if col not in {tempcol_1, tempcol_2}
        ]
        return df.select(ordered_cols)

    elif step["operation"] == "Remove Outliers":
        return remove_outlier_by_IQR(df, numericCols)
    else:
        print(
            f"error: Operation not defined in All_Column_Operations function for {step['column']} column: {step['operation']} \n"
        )
        return df


def Column_Operations(df, step):
    column = step["column"]
    if step["operation"] == "Drop Null":
        return df.dropna(subset=column)

    elif step["operation"] == "Drop Duplicates":
        return df.dropDuplicates(subset=column)

    elif step["operation"] == "Drop Column":
        return df.drop(column)

    elif step["operation"] == "Fill 0":
        return df.fillna(0, subset=column)

    elif step["operation"] in ["Fill mean", "Fill Mode", "Fill Median"]:
        strategy = (
            "mean"
            if step["operation"] == "Fill mean"
            else "mode" if step["operation"] == "Fill Mode" else "median"
        )
        imputer = Imputer(inputCol=column, outputCol=column, strategy=strategy)
        return imputer.fit(df).transform(df)

    elif step["operation"] == "Fill Unknown":
        return df.fillna("Unknown", subset=column)

    elif step["operation"] == "Fill False":
        df = df.fillna(False, subset=column)

    elif step["operation"] in [
        "L1 Norm",
        "L2 Norm",
        "L inf Norm",
        "Min-Max",
        "Z-score",
    ]:
        return normalize_column(df, column, step["operation"])

    elif step["operation"] == "Remove Outliers":
        return remove_outlier_by_IQR(df, column)

    elif step["operation"] == "Log":
        return df.withColumn(column, F.log(F.col(column)))

    elif step["operation"] == "Square":
        return df.withColumn(column, (F.col(column)) * 2)

    elif step["operation"] == "Square Root":
        return df.withColumn(column, F.sqrt(F.col(column)))

    elif step["operation"] == "Label Encoding":

        if df.filter(col(column).isNull()).count() > 0:
            print(f"error: Null values found in {column} column for Label Encoding")
            return df

        temp_col1 = get_temp_col("features")
        indexer = StringIndexer(inputCol=column, outputCol=temp_col1)
        df = indexer.fit(df).transform(df)
        return df.withColumn(column, col(temp_col1)).drop(temp_col1)

    elif step["operation"] == "One Hot Encoding":
        # this gives sparse vector, which if not compatible with ML model then have to encode in dense vectors

        if df.filter(col(column).isNull()).count() > 0:
            print(f"error: Null values found in {column} column for One Hot Encoding")
            return df

        temp_col1 = get_temp_col("features")
        # check if column is string type
        if isinstance(df.schema[column].dataType, StringType):
            indexer = StringIndexer(inputCol=column, outputCol=temp_col1)
            df = indexer.fit(df).transform(df)
            df = df.withColumn(column, col(temp_col1)).drop(temp_col1)

        encoder = OneHotEncoder(inputCol=column, outputCol=temp_col1)
        df = encoder.fit(df).transform(df)
        return df.withColumn(column, col(temp_col1)).drop(temp_col1)

    else:
        print(
            f"error: Operation not defined in Column_Operations function for {step['column']} column: {step['operation']} \n"
        )
        return df

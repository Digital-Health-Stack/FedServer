import math
from functools import reduce
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    StandardScaler,
)


def get_temp_col(base: str) -> str:
    return f"{base}_{uuid4().hex[:8]}"


def remove_outlier_by_IQR(dataframe, columns, factor=1.5):
    if isinstance(columns, str):
        columns = [columns]
    mask = pd.Series(True, index=dataframe.index)
    for column in columns:
        if column not in dataframe.columns:
            continue
        s = dataframe[column]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        mask &= s.between(lower_bound, upper_bound, inclusive="both")
    return dataframe.loc[mask].copy()


def normalize_column(df, column_name, method):
    df = df.copy()
    s = df[column_name]
    if not pd.api.types.is_numeric_dtype(s):
        print(
            f"Unsupported or non-numeric column for normalization: {column_name} ({method})"
        )
        return df

    if method == "Min-Max":
        min_val, max_val = s.min(), s.max()
        if max_val == min_val:
            df[column_name] = 0.0
        else:
            df[column_name] = (s - min_val) / (max_val - min_val)

    elif method == "Z-score":
        mean_val, std_val = s.mean(), s.std()
        if std_val == 0 or pd.isna(std_val):
            df[column_name] = 0.0
        else:
            df[column_name] = (s - mean_val) / std_val

    elif method == "L1 Norm":
        abs_sum = s.abs().sum()
        if abs_sum == 0:
            df[column_name] = 0.0
        else:
            df[column_name] = s / abs_sum

    elif method == "L2 Norm":
        squared_sum = (s.astype(float) ** 2).sum()
        if squared_sum == 0:
            df[column_name] = 0.0
        else:
            df[column_name] = s / math.sqrt(squared_sum)

    elif method == "L inf Norm":
        abs_max = s.abs().max()
        if abs_max == 0:
            df[column_name] = 0.0
        else:
            df[column_name] = s / abs_max

    else:
        print(
            f"Unsupported normalization method: {method} for the column {column_name}"
        )
    return df


def All_Column_Operations(df, step, numericCols, allCols):
    df = df.copy()
    allCols = [c for c in allCols if c in df.columns]
    numericCols = [c for c in numericCols if c in df.columns]

    if step["operation"] == "Drop Null":
        return df.dropna(subset=allCols)

    if step["operation"] == "Fill 0 Unknown False":
        for c in allCols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(0)
            elif pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].fillna(False)
            else:
                df[c] = df[c].fillna("unknown")
        return df

    if step["operation"] == "Fill Mean":
        if not numericCols:
            return df
        imp = SimpleImputer(strategy="mean")
        df[numericCols] = imp.fit_transform(df[numericCols])
        return df

    if step["operation"] == "Fill Median":
        if not numericCols:
            return df
        imp = SimpleImputer(strategy="median")
        df[numericCols] = imp.fit_transform(df[numericCols])
        return df

    if step["operation"] == "Drop Duplicates":
        return df.drop_duplicates()

    if step["operation"] in [
        "L1 Norm",
        "L2 Norm",
        "L inf Norm",
        "Min-Max",
        "Z-score",
    ]:
        if not numericCols:
            return df
        M = df[numericCols].astype(float).values
        if step["operation"] == "L1 Norm":
            out = Normalizer(norm="l1").fit_transform(M)
        elif step["operation"] == "L2 Norm":
            out = Normalizer(norm="l2").fit_transform(M)
        elif step["operation"] == "L inf Norm":
            out = Normalizer(norm="max").fit_transform(M)
        elif step["operation"] == "Min-Max":
            out = MinMaxScaler().fit_transform(M)
        else:
            out = StandardScaler(with_mean=False, with_std=True).fit_transform(M)
        df[numericCols] = out
        return df

    if step["operation"] == "Remove Outliers":
        return remove_outlier_by_IQR(df, numericCols)

    print(
        f"error: Operation not defined in All_Column_Operations function for {step['column']} column: {step['operation']} \n"
    )
    return df


def Column_Operations(df, step):
    df = df.copy()
    column = step["column"]

    if step["operation"] == "Drop Null":
        return df.dropna(subset=[column])

    if step["operation"] == "Drop Duplicates":
        return df.drop_duplicates(subset=[column])

    if step["operation"] == "Drop Column":
        return df.drop(columns=[column], errors="ignore")

    if step["operation"] == "Fill 0":
        return df.fillna({column: 0})

    if step["operation"] in ["Fill mean", "Fill Mode", "Fill Median"]:
        strategy = (
            "mean"
            if step["operation"] == "Fill mean"
            else "most_frequent"
            if step["operation"] == "Fill Mode"
            else "median"
        )
        imp = SimpleImputer(strategy=strategy)
        df[column] = imp.fit_transform(df[[column]]).ravel()
        return df

    if step["operation"] == "Fill Unknown":
        return df.fillna({column: "Unknown"})

    if step["operation"] == "Fill False":
        return df.fillna({column: False})

    if step["operation"] in [
        "L1 Norm",
        "L2 Norm",
        "L inf Norm",
        "Min-Max",
        "Z-score",
    ]:
        return normalize_column(df, column, step["operation"])

    if step["operation"] == "Remove Outliers":
        return remove_outlier_by_IQR(df, [column])

    if step["operation"] == "Log":
        return df.assign(**{column: np.log(pd.to_numeric(df[column], errors="coerce"))})

    if step["operation"] == "Square":
        return df.assign(**{column: np.square(pd.to_numeric(df[column], errors="coerce"))})

    if step["operation"] == "Square Root":
        return df.assign(
            **{column: np.sqrt(pd.to_numeric(df[column], errors="coerce"))}
        )

    if step["operation"] == "Label Encoding":
        if df[column].isna().any():
            print(f"error: Null values found in {column} column for Label Encoding")
            return df
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        return df

    if step["operation"] == "One Hot Encoding":
        if df[column].isna().any():
            print(f"error: Null values found in {column} column for One Hot Encoding")
            return df
        if df[column].dtype == object or str(df[column].dtype) == "string":
            le = LabelEncoder()
            enc_col = get_temp_col("le")
            df[enc_col] = le.fit_transform(df[column].astype(str))
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            oh = ohe.fit_transform(df[[enc_col]])
            prefix = f"{column}_oh"
            for j in range(oh.shape[1]):
                df[f"{prefix}_{j}"] = oh[:, j]
            df = df.drop(columns=[column, enc_col])
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            oh = ohe.fit_transform(df[[column]])
            prefix = f"{column}_oh"
            for j in range(oh.shape[1]):
                df[f"{prefix}_{j}"] = oh[:, j]
            df = df.drop(columns=[column])
        return df

    print(
        f"error: Operation not defined in Column_Operations function for {step['column']} column: {step['operation']} \n"
    )
    return df

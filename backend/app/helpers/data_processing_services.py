import os
import tempfile
import time
import uuid
from typing import Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from helpers.aws_services import S3Services
from helpers.local_storage_services import LocalStorageManager
from helpers.processing_helper_functions import All_Column_Operations, Column_Operations

load_dotenv()


def serialize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    try:
        if obj is not None and pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    return obj


class DataProcessingManager:
    def __init__(self):
        self.storage = LocalStorageManager()
        self.s3 = S3Services()
        self.merge_temp = os.getenv("MERGE_TEMP_DIRECTORY", "_merge_temp")

    def _dtype_name(self, dtype) -> str:
        if pd.api.types.is_integer_dtype(dtype):
            return "IntegerType()"
        if pd.api.types.is_float_dtype(dtype):
            return "DoubleType()"
        if pd.api.types.is_bool_dtype(dtype):
            return "BooleanType()"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "TimestampType()"
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            return "StringType()"
        if pd.api.types.is_categorical_dtype(dtype):
            return "StringType()"
        return str(dtype)

    def _get_overview(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {"message": "Dataset not found."}

        num_rows = len(df)
        column_stats = []

        for column in df.columns:
            try:
                s = df[column]
                col_type = self._dtype_name(s.dtype)
                stats: dict = {
                    "name": column,
                    "type": col_type,
                    "entries": int(s.notna().sum()),
                    "nullCount": int(s.isna().sum()),
                }

                if pd.api.types.is_numeric_dtype(s):
                    sn = pd.to_numeric(s, errors="coerce").dropna()
                    if len(sn) == 0:
                        column_stats.append(stats)
                        continue
                    stats["mean"] = float(sn.mean())
                    stats["stddev"] = float(sn.std()) if len(sn) > 1 else 0.0
                    stats["min"] = float(sn.min())
                    stats["max"] = float(sn.max())
                    stats["uniqueCount"] = int(sn.nunique())
                    q1, med, q3 = sn.quantile([0.25, 0.5, 0.75])
                    stats["quartiles"] = {
                        "Q1": float(q1),
                        "median": float(med),
                        "Q3": float(q3),
                        "IQR": float(q3 - q1),
                    }
                    min_val, max_val = stats["min"], stats["max"]
                    if max_val > min_val:
                        bins = np.linspace(min_val, max_val, 11)
                        counts, edges = np.histogram(sn.values, bins=bins)
                        stats["histogram"] = {
                            "bins": edges.tolist(),
                            "counts": counts.tolist(),
                        }
                    else:
                        stats["histogram"] = {"bins": [min_val], "counts": [len(sn)]}

                elif pd.api.types.is_string_dtype(s) or s.dtype == object:
                    row0 = s.dropna()
                    first = row0.iloc[0] if len(row0) else None
                    if isinstance(first, (list, np.ndarray)):
                        def flatten_all(x):
                            if isinstance(x, (list, np.ndarray)):
                                for i in x:
                                    yield from flatten_all(i)
                            else:
                                yield x

                        shape = []
                        temp = first
                        while isinstance(temp, list):
                            shape.append(len(temp))
                            if len(temp) == 0:
                                break
                            temp = temp[0] if isinstance(temp[0], list) else None
                        stats["Shape"] = tuple(shape) if shape else None

                        lens = s.dropna().map(
                            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
                        )
                        if len(lens):
                            stats["LengthStats"] = {
                                "min": int(lens.min()),
                                "max": int(lens.max()),
                                "mean": float(lens.mean()),
                                "std": float(lens.std()) if len(lens) > 1 else 0.0,
                            }
                        flat_sample = []
                        if first is not None and isinstance(first, list):
                            flat_sample = list(flatten_all(first))
                        if flat_sample and isinstance(flat_sample[0], (int, float, np.number)):
                            num_samples = int(np.minimum(num_rows * 0.2, 100000))
                            collected = []
                            for v in s.dropna().head(max(1, num_rows // 10 or 1)):
                                collected.extend(list(flatten_all(v)))
                            sampled = collected[:num_samples]
                            if sampled:
                                arr_np = np.array(sampled, dtype=float)
                                stats["sampleSize"] = f"{len(sampled)} samples"
                                stats["valueStats"] = {
                                    "min": float(np.min(arr_np)),
                                    "max": float(np.max(arr_np)),
                                    "mean": float(np.mean(arr_np)),
                                    "std": float(np.std(arr_np)),
                                    "median": float(np.median(arr_np)),
                                    "sparsity": float(np.mean(arr_np == 0)),
                                }
                            else:
                                stats["valueStats"] = None
                        else:
                            stats["valueStats"] = (
                                "Not numeric" if flat_sample else "Not detected"
                            )
                    else:
                        vc = s.dropna().astype(str).value_counts().head(10)
                        stats["uniqueCount"] = int(s.nunique(dropna=True))
                        stats["topCategories"] = [
                            {
                                "value": (
                                    (val[:50] + "...")
                                    if isinstance(val, str) and len(val) > 50
                                    else val
                                ),
                                "count": int(cnt),
                            }
                            for val, cnt in vc.items()
                        ]

                else:
                    pass

                column_stats.append(stats)
            except Exception as e:
                print(f"Error processing column {column}: {e}")
                continue

        overview = {
            "numRows": num_rows,
            "numColumns": len(df.columns),
            "columnStats": column_stats,
        }
        return serialize_for_json(overview)

    async def create_new_dataset(self, filename: str, filetype: str):
        try:
            print(f"in create_new_dataset {filename} is {filetype}")
            src = self.storage.get_path(filename)
            if not os.path.exists(src):
                raise FileNotFoundError(f"Uploaded file not found: {filename}")

            write_filename = filename
            if filetype == "csv":
                df = pd.read_csv(src)
                write_filename = filename.replace(".csv", ".parquet")
                dest = self.storage.get_path(write_filename)
                df.to_parquet(dest, index=False)
                try:
                    os.remove(src)
                except OSError as e:
                    print(f"Warning: could not remove uploaded CSV {filename}: {e}")
            elif filetype == "parquet":
                df = pd.read_parquet(src)
                dest = self.storage.get_path(write_filename)
                df.to_parquet(dest, index=False)
            else:
                print("Unsupported file type for creating new dataset.")
                return {"message": "Unsupported file type."}

            dataset_overview = self._get_overview(df)
            dataset_overview["filename"] = write_filename
            return dataset_overview
        except Exception as e:
            print(f"Error creating new dataset: {e}")
            raise e

    async def preprocess_data(
        self, dataset_type: str, filename: str, operations: List[dict]
    ):
        try:
            path = self.storage.get_path(filename)
            print(f"Starting preprocessing for {path} (dataset_type={dataset_type})...")
            df = pd.read_parquet(path)

            All_Columns = list(df.columns)
            numericCols = [
                c
                for c in All_Columns
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]

            t1 = time.time()
            for step in operations:
                if step["operation"] == "Exclude from All Columns list":
                    if step["column"] in All_Columns:
                        All_Columns.remove(step["column"])
                    if step["column"] in numericCols:
                        numericCols.remove(step["column"])
                elif step["column"] == "All Columns":
                    try:
                        df = All_Column_Operations(df, step, numericCols, All_Columns)
                    except Exception as e:
                        print(
                            f"error: Error in {step['operation']} operation for {step['column']} column: {str(e)} \n"
                        )
                else:
                    try:
                        df = Column_Operations(df, step)
                    except Exception as e:
                        print(
                            f"error: Error in {step['operation']} operation for {step['column']} column: {str(e)} \n"
                        )

            newfilename = f"{filename}_{uuid.uuid4().hex}.parquet"
            out = self.storage.get_path(newfilename)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            df.to_parquet(out, index=False)

            print(
                f"Preprocessed dataset saved to: {out} and time taken: ",
                time.time() - t1,
            )

            overview = self._get_overview(df)
            overview["filename"] = newfilename
            return overview
        except Exception as e:
            print(f"Error in preprocessing dataset: {e}")
            raise e

    async def merge_s3_into_local_dataset(
        self, s3_path: str, parent_filename: str, session_id: int
    ):
        _ = session_id
        merge_rel = f"{self.merge_temp}/{parent_filename}"
        merge_abs = self.storage.get_path(merge_rel)
        local_parent = self.storage.get_path(parent_filename)

        try:
            print(f"reading env variables...merge dir: {self.merge_temp}")
            if not os.path.exists(local_parent):
                raise FileNotFoundError(f"Parent dataset not found: {parent_filename}")

            hdfs_df = pd.read_parquet(local_parent)

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                self.s3.download_uri_to_path(s3_path, tmp_path)
                s3_df = pd.read_parquet(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            if len(hdfs_df) == 0 or len(s3_df) == 0:
                msg = f"File {local_parent} or S3 object is empty, skipping merge."
                print(msg)
                raise Exception(msg)

            cols = sorted(set(hdfs_df.columns) | set(s3_df.columns))
            hdfs_df = hdfs_df.reindex(columns=cols)
            s3_df = s3_df.reindex(columns=cols)
            merged = pd.concat([hdfs_df, s3_df], ignore_index=True)

            os.makedirs(os.path.dirname(merge_abs), exist_ok=True)
            merged.to_parquet(merge_abs, index=False)
            print(f"Merged dataset saved to: {merge_abs}")

            if self.storage.path_exists(parent_filename):
                await self.storage.delete_file(parent_filename)

            await self.storage.rename_file_or_folder(merge_rel, parent_filename)
            print(f"Moved merged dataset to: {self.storage.get_path(parent_filename)}")

            overview = self._get_overview(merged)
            overview["filename"] = parent_filename
            return overview

        except Exception as e:
            print(f"Error Merging S3 dataset: {e}")
            if self.storage.path_exists(merge_rel):
                try:
                    await self.storage.delete_file(merge_rel)
                except Exception as cleanup_err:
                    print(f"Cleanup merge temp failed: {cleanup_err}")
            raise e

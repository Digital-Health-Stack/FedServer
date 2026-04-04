import os
import shutil

import numpy as np
import pandas as pd

from services.local_storage import LocalStorageManager


def reshape_image(img_array):
    img_array = np.stack([np.stack(row, axis=0) for row in img_array], axis=0)
    return img_array.astype(np.float32)


def process_parquet_and_save_xy(
    filename: str,
    session_id: str,
    input_columns: list,
    output_column: list,
):
    """
    Load parquet files from local storage, combine them,
    extract X and Y arrays, save them under ./data.

    Args:
        filename: Storage entry (folder or file) name under LOCAL_STORAGE_DIR
        session_id: Unique session ID for temp file management
        input_columns: Feature columns
        output_column: Target column(s)

    Returns:
        None (saves X/Y npy files under ./data)
    """

    local_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(local_dir, exist_ok=True)

    temp_download_dir = os.path.join(local_dir, f"temp_{session_id}")
    os.makedirs(temp_download_dir, exist_ok=True)

    storage = LocalStorageManager()
    storage.copy_storage_entry_to_local(filename, temp_download_dir)

    combined_df = None
    parquet_files = []

    for root, _, files in os.walk(temp_download_dir):
        for file in files:
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                parquet_files.append(file_path)

                df = pd.read_parquet(file_path)
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

    shutil.rmtree(temp_download_dir)

    if not parquet_files or combined_df is None:
        raise Exception("No parquet files found in the downloaded folder")

    print(f"Combined DataFrame Shape: {combined_df.shape}")
    print(f"DataFrame Column Labels: {combined_df.columns.tolist()}")

    missing_cols = [col for col in output_column if col not in combined_df.columns]
    if missing_cols:
        raise Exception(f"Output column(s) not found in the DataFrame: {missing_cols}")

    print(combined_df.dtypes)
    print("Check head", combined_df.head())

    X = combined_df[input_columns].values
    Y = combined_df[output_column].values

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    X_filename = os.path.join(local_dir, f"X_{session_id}.npy")
    Y_filename = os.path.join(local_dir, f"Y_{session_id}.npy")

    np.save(X_filename, X)
    np.save(Y_filename, Y)

    return

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
from helpers.spark_services import SparkSessionManager
from helpers.aws_services import S3Services
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

load_dotenv()

from crud.training_data_transfer_crud import (
    create_transfer,
    get_all_transfers,
    get_pending_transfers,
    get_transfer_details,
    approve_transfer,
    delete_transfer,
    get_transfer_mini_details,
)

from crud.datasets_crud import update_dataset_stats, create_dataset
from schemas.dataset import DatasetCreate
from schemas.training_data_transfer import (
    TransferCreate,
    TransferListItem,
    TransferDetails,
)

from utility.db import get_db

spark_client = SparkSessionManager()
s3_client = S3Services()

executor = ThreadPoolExecutor(max_workers=os.cpu_count())

qpd_router = APIRouter(tags=["QPD"])

QPD_DATASET_DIR_ON_S3 = os.getenv("QPD_DATASET_DIR_ON_S3")


def handle_error(result):
    # --------------------------------------------------------------------------------------------
    # Handle the error if the result is a dictionary with an "error" key
    # this is required to log the proper error at the api-endpoint file itself not in crud file
    # --------------------------------------------------------------------------------------------
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


async def merge_s3_file_to_hdfs(transfer_id: int):
    try:
        db = next(get_db())
        result = get_transfer_mini_details(db, transfer_id)
        handle_error(result)
        print("starting the merging process...")

        # Merge S3 file with parent file on HDFS and delete the file on S3
        overview = await spark_client.merge_s3_file_to_hdfs(
            result.data_path, result.parent_filename, result.federated_session_id
        )

        # new_dataset = DatasetCreate(
        #     filename=overview["filename"],
        #     description=f"Merged {result.parent_filename} with QPD data of {result.federated_session_id}",
        #     datastats=overview
        # )

        # crud_result = create_dataset(db, dataset=new_dataset)
        # if isinstance(crud_result, dict) and "error" in crud_result:
        #     raise HTTPException(status_code=400, detail=crud_result["error"])

        result = update_dataset_stats(db, overview["filename"], overview)
        # result = update_dataset_stats(db, result.parent_filename, overview)
        handle_error(result)
        print("DB updated with merged dataset stats")

        result = approve_transfer(db, transfer_id)
        handle_error(result)
        print("Transfer approved successfully")

        # e.g. s3a://qpd-data/temp/4934bd27-c155-4303-b386-64b7cd030fe5_health_client.parquet
        s3_filename = result.data_path.split("/")[-1]
        s3_client.delete_folder(f"{QPD_DATASET_DIR_ON_S3}/{s3_filename}")
        print(f"file {s3_filename} deleted successfully from S3")

        return {"message": "Approved successfully"}
    except Exception as e:
        print(f"Error during approval process: {e}")
        return {"error": str(e)}


@qpd_router.post("/create-transferred-data")
def create_data_transfer(transfer_data: TransferCreate, db: Session = Depends(get_db)):
    try:
        result = create_transfer(db, transfer_data)
        result = handle_error(result)
        return result.as_dict()
    except Exception as e:
        print(f"Error creating transfer: {e}")
        return {"error": str(e)}


@qpd_router.get("/list-transferred-data", response_model=List[TransferListItem])
def list_all_transfers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    try:
        transfers = get_pending_transfers(db, skip=skip, limit=limit)
        return handle_error(transfers)
    except Exception as e:
        print(f"Error listing pending transfers: {e}")
        return {"error": str(e)}


@qpd_router.get(
    "/transferred-data-overview/{transfer_id}", response_model=TransferDetails
)
def get_transfer_overview(transfer_id: int, db: Session = Depends(get_db)):
    try:
        details = get_transfer_details(db, transfer_id)
        return handle_error(details)
    except Exception as e:
        print(f"Error fetching transfer details: {e}")
        return {"error": str(e)}


@qpd_router.delete("/delete-transferred-data/{transfer_id}")
def delete_transfer_record(transfer_id: int, db: Session = Depends(get_db)):
    try:
        result = delete_transfer(db, transfer_id)
        return handle_error(result)
    except Exception as e:
        print(f"Error deleting transfer: {e}")
        return {"error": str(e)}


@qpd_router.post("/approve-transferred-data/{transfer_id}")
def approve_transfer_record(transfer_id: int):
    # don't do try/except here as this will run in a separate thread
    executor.submit(asyncio.run, merge_s3_file_to_hdfs(transfer_id))
    print("Approval process started for transfer ID:", transfer_id)
    return {"message": "Approval process started"}

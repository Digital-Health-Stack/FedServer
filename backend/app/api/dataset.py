from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi import Request
from sqlalchemy.orm import Session
from typing import List
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from schemas.dataset import (
    DatasetCreate,
    RawDatasetListResponse,
    DatasetListResponse,
    Operation,
    DatasetUpdate,
)

from crud.datasets_crud import (
    create_raw_dataset,
    delete_raw_dataset,
    rename_raw_dataset,
    list_raw_datasets,
    get_raw_dataset_stats,
    create_dataset,
    delete_dataset,
    rename_dataset,
    list_datasets,
    get_dataset_stats,
    get_data_filename_by_id,
    get_raw_data_filename_by_id,
    edit_dataset_details,
    edit_raw_dataset_details,
    handle_file_renaming_during_processing
)

from utility.db import get_db
from helpers.hdfs_services import HDFSServiceManager
from helpers.spark_services import SparkSessionManager
from dotenv import load_dotenv

load_dotenv()

executor = ThreadPoolExecutor(max_workers=os.cpu_count())

dataset_router = APIRouter(tags=["Dataset"])

HDFS_RAW_DATASETS_DIR = os.getenv("HDFS_RAW_DATASETS_DIR")
HDFS_PROCESSED_DATASETS_DIR = os.getenv("HDFS_PROCESSED_DATASETS_DIR")
RECENTLY_UPLOADED_DATASETS_DIR = os.getenv("RECENTLY_UPLOADED_DATASETS_DIR")

hdfs_client = HDFSServiceManager()
spark_client = SparkSessionManager()

###################### Background processing tasks ######################
async def process_create_dataset(filename: str, filetype: str):
    db = next(get_db())
    try:
        source_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}"
        processing_path = f"{source_path}__PROCESSING__"
        
        await hdfs_client.rename_file_or_folder(source_path, processing_path)
        dataset_overview = await spark_client.create_new_dataset(f"{filename}__PROCESSING__", filetype)
        description = f"Raw dataset created from {filename}"

        dataset_obj = DatasetCreate(filename=filename, description=description, datastats=dataset_overview)

        # Create raw dataset entry
        crud_result = create_raw_dataset(db, dataset_obj)
        if isinstance(crud_result, dict) and "error" in crud_result:
            raise HTTPException(status_code=400, detail=crud_result["error"])
        
        await hdfs_client.rename_file_or_folder(processing_path, source_path)
        return {"message": "Dataset created successfully"}
    except Exception as e:
        await hdfs_client.rename_file_or_folder(processing_path, source_path, ignore_missing=True)
        print("Error in processing the data is: ", str(e))
        return {"error": str(e)}
    finally:
        db.close()

    
async def process_preprocessing(directory: str, filename: str, operations: List[Operation]):
    db = next(get_db())
    try:
        processing_path = f"{directory}/{filename}__PROCESSING__"
        await hdfs_client.rename_file_or_folder(f"{directory}/{filename}", processing_path)
        
        renaming_result = handle_file_renaming_during_processing(db, filename, f"{filename}__PROCESSING__", directory)
        if isinstance(renaming_result, dict) and "error" in renaming_result:
            raise HTTPException(status_code=400, detail=renaming_result["error"])
        
        # Process data and get new filename
        processed_info = await spark_client.preprocess_data(
            directory, 
            f"{filename}__PROCESSING__", 
            operations
        )

        print("Check:: in processing fn- if error goes to except block: ", processed_info)
        # Create new dataset entry
        new_dataset = DatasetCreate(
            filename=processed_info["filename"],
            description=f"Processed version of {filename}",
            datastats=processed_info
        )

        crud_result = create_dataset(db, dataset=new_dataset)
        if isinstance(crud_result, dict) and "error" in crud_result:
            raise HTTPException(status_code=400, detail=crud_result["error"])
        
        await hdfs_client.rename_file_or_folder(processing_path, f"{directory}/{filename}")

        renaming_result = handle_file_renaming_during_processing(db, f"{filename}__PROCESSING__", filename, directory)
        if isinstance(renaming_result, dict) and "error" in renaming_result:
            raise HTTPException(status_code=400, detail=renaming_result["error"])
        return {"message": "Preprocessing completed successfully"}
        
    except Exception as e:
        await hdfs_client.rename_file_or_folder(processing_path, f"{directory}/{filename}", ignore_missing=True)
        handle_file_renaming_during_processing(db, f"{filename}__PROCESSING__", filename, directory)
        print("Error in preprocessing the data is: ", str(e))
        return {"error": str(e)}
    finally:
        db.close()

######################## Dataset Routes #######################

@dataset_router.get("/preprocessing", summary="Test server connection")
def hello_server():
    return {"message": "Preprocessing router operational"}



############ Raw Dataset Management Routes
@dataset_router.get("/list-raw-datasets", response_model=List[RawDatasetListResponse])
def list_raw_datasets_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    try:
        result = list_raw_datasets(db, skip=skip, limit=limit)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print("Error in listing raw datasets: ", str(e))
        return {"error": str(e)}

@dataset_router.get("/raw-dataset-details/{filename}", response_model=dict)
def get_raw_dataset_overview(filename: str, db: Session = Depends(get_db)):
    try:
        result = get_raw_dataset_stats(db, filename=filename)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        print("Error in getting raw dataset overview: ", str(e))
        return {"error": str(e)}

@dataset_router.put("/rename-raw-dataset-file")
async def rename_raw_dataset_file(
    old_file_name: str = Query(...),
    new_file_name: str = Query(...),
    db: Session = Depends(get_db)
):
    try:
        # Rename file on HDFS
        await hdfs_client.rename_file_or_folder(
            f"{HDFS_RAW_DATASETS_DIR}/{old_file_name}",
            f"{HDFS_RAW_DATASETS_DIR}/{new_file_name}"
        )

        result = rename_raw_dataset(db, old_file_name, new_file_name)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        print("Error in renaming raw dataset: ", str(e))
        return {"error": str(e)}
    
@dataset_router.put("/edit-raw-dataset-details")
async def edit_raw_dataset(
    newdetails: DatasetUpdate,
    db: Session = Depends(get_db)
):
    try:
        # get dataset name and edit on hdfs
        old_file_name = get_raw_data_filename_by_id(db, newdetails.dataset_id)
        if isinstance(old_file_name, dict) and "error" in old_file_name:
            raise HTTPException(status_code=404, detail=old_file_name["error"])
        
        if old_file_name != newdetails.filename:
            await hdfs_client.rename_file_or_folder(
                f"{HDFS_RAW_DATASETS_DIR}/{old_file_name}",
                f"{HDFS_RAW_DATASETS_DIR}/{newdetails.filename}"
            )
        
        result = edit_raw_dataset_details(db, newdetails)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {"message": "Raw dataset details updated successfully"}
    except Exception as e:
        print("Error in editing raw dataset details: ", str(e))
        return {"error": str(e)}
    
@dataset_router.delete("/delete-raw-dataset-file")
async def delete_raw_dataset_file(
    dataset_id: str = Query(...),
    db: Session = Depends(get_db)
):
    try:
        # delete file on HDFS
        filename = get_raw_data_filename_by_id(db, dataset_id)
        if isinstance(filename, dict) and "error" in filename:
            raise HTTPException(status_code=404, detail=filename["error"])
        
        await hdfs_client.delete_file_from_hdfs(HDFS_RAW_DATASETS_DIR, filename)

        result = delete_raw_dataset(db, dataset_id)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print("Error in deleting raw dataset: ", str(e))
        return {"error": str(e)}

@dataset_router.post("/create-new-dataset", status_code=status.HTTP_202_ACCEPTED)
async def create_new_dataset(request: Request):
    data = await request.json()
    filename = data.get("fileName")
    filetype = filename.split(".")[-1].lower()
    
    if filetype not in ["csv", "parquet"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported formats: CSV, Parquet"
        )
    
    executor.submit(
        asyncio.run, 
        process_create_dataset(filename, filetype)
    )
    return {"message": "Dataset processing started"}



############ Processed Dataset Management Routes
@dataset_router.get("/list-datasets", response_model=List[DatasetListResponse])
def list_datasets_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    try:
        result = list_datasets(db, skip=skip, limit=limit)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print("Error in listing processed datasets: ", str(e))
        return {"error": str(e)}

@dataset_router.get("/dataset-details/{filename}", response_model=dict)
def get_dataset_overview(filename: str, db: Session = Depends(get_db)):
    try:
        result = get_dataset_stats(db, filename=filename)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        print("Error in getting processed dataset overview: ", str(e))
        return {"error": str(e)}

@dataset_router.put("/rename-dataset-file")
async def rename_processed_dataset_file(
    dataset_id: int = Query(...),
    new_name: str = Query(...),
    db: Session = Depends(get_db)
):
    try:
        # get dataset name and rename on hdfs
        old_file_name = get_data_filename_by_id(db, dataset_id)
        if isinstance(old_file_name, dict) and "error" in old_file_name:
            raise HTTPException(status_code=404, detail=old_file_name["error"])
        
        await hdfs_client.rename_file_or_folder(
            f"{HDFS_PROCESSED_DATASETS_DIR}/{old_file_name}",
            f"{HDFS_PROCESSED_DATASETS_DIR}/{new_name}"
        )

        # don't do it by ID, do it by filename (the function is reffered on other places too)
        result = rename_dataset(db, old_file_name, new_name)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print("Error in renaming processed dataset: ", str(e))
        return {"error": str(e)}

@dataset_router.put("/edit-dataset-details")
async def edit_raw_dataset(
    newdetails: DatasetUpdate,
    db: Session = Depends(get_db)
):
    try:
        # get dataset name and edit on hdfs
        old_file_name = get_data_filename_by_id(db, newdetails.dataset_id)
        if isinstance(old_file_name, dict) and "error" in old_file_name:
            raise HTTPException(status_code=404, detail=old_file_name["error"])
        
        if old_file_name != newdetails.filename:
            await hdfs_client.rename_file_or_folder(
                f"{HDFS_PROCESSED_DATASETS_DIR}/{old_file_name}",
                f"{HDFS_PROCESSED_DATASETS_DIR}/{newdetails.filename}"
            )

        result = edit_dataset_details(db, newdetails)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {"message": "Dataset details updated successfully"}
    except Exception as e:
        print("Error in editing dataset details: ", str(e))
        return {"error": str(e)}
    
@dataset_router.delete("/delete-dataset-file")
async def delete_processed_dataset_file(
    dataset_id: int = Query(...),
    db: Session = Depends(get_db)
):
    try:
        # delete file on HDFS
        filename = get_data_filename_by_id(db, dataset_id)
        if isinstance(filename, dict) and "error" in filename:
            raise HTTPException(status_code=404, detail=filename["error"])
        
        await hdfs_client.delete_file_from_hdfs(HDFS_PROCESSED_DATASETS_DIR, filename)

        result = delete_dataset(db, dataset_id)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print("Error in deleting processed dataset: ", str(e))
        return {"error": str(e)}

@dataset_router.post("/preprocess-dataset", status_code=status.HTTP_202_ACCEPTED)
async def preprocess_dataset_endpoint(request: Request):
    data = await request.json()
    executor.submit(
        asyncio.run,
        process_preprocessing(
            data["directory"],
            data["filename"],
            data["operations"]
        )
    )
    return {"message": "Preprocessing initiated"}

@dataset_router.get("/list-recent-uploads")   
async def list_recent_uploads():
    return await hdfs_client.list_recent_uploads()
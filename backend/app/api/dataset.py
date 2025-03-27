from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi import Request, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from schemas.dataset import (
    DatasetCreate,
    DatasetResponse,
    TaskCreate,
    TaskResponse,
    BenchmarkCreate,
    BenchmarkResponse,
    RawDatasetListResponse,
    DatasetListResponse,
    Operation
)
from helpers.datasets_crud import (
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
    create_task,
    delete_task,
    create_benchmark,
    delete_benchmark,
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

# Helper function to handle CRUD results
def handle_crud_result(result):
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

###################### Background processing tasks ######################
async def process_create_dataset(filename: str, filetype: str):
    db = next(get_db())
    try:
        source_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}"
        processing_path = f"{source_path}__PROCESSING__"
        
        await hdfs_client.rename_file_or_folder(source_path, processing_path)
        dataset_overview = await spark_client.create_new_dataset(f"{filename}__PROCESSING__", filetype)
        
        # Create raw dataset entry
        crud_result = create_raw_dataset(db, filename=filename, datastats=dataset_overview)
        handle_crud_result(crud_result)
        
        await hdfs_client.rename_file_or_folder(processing_path, source_path)
    except Exception as e:
        await hdfs_client.rename_file_or_folder(processing_path, source_path, ignore_missing=True)
        raise e
    finally:
        db.close()

async def process_preprocessing(directory: str, filename: str, operations: List[Operation]):
    db = next(get_db())
    try:
        processing_path = f"{directory}/{filename}__PROCESSING__"
        await hdfs_client.rename_file_or_folder(f"{directory}/{filename}", processing_path)
        
        # Process data and get new filename
        processed_info = await spark_client.preprocess_data(
            directory, 
            f"{filename}__PROCESSING__", 
            operations
        )
        
        # Create new dataset entry
        new_dataset = DatasetCreate(
            filename=processed_info["fileName"],
            description=f"Processed version of {filename}",
            datastats=processed_info["stats"]
        )
        crud_result = create_dataset(db, dataset=new_dataset)
        handle_crud_result(crud_result)
        
        await hdfs_client.rename_file_or_folder(processing_path, f"{directory}/{filename}")
    except Exception as e:
        await hdfs_client.rename_file_or_folder(processing_path, f"{directory}/{filename}", ignore_missing=True)
        raise e
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
    result = list_raw_datasets(db, skip=skip, limit=limit)
    return handle_crud_result(result)

@dataset_router.get("/raw-dataset-details/{filename}", response_model=dict)
def get_raw_dataset_overview(filename: str, db: Session = Depends(get_db)):
    result = get_raw_dataset_stats(db, filename=filename)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@dataset_router.put("/rename-raw-dataset-file")
def rename_raw_dataset_file(
    old_file_name: str = Query(...),
    new_file_name: str = Query(...),
    db: Session = Depends(get_db)
):
    result = rename_raw_dataset(db, old_file_name, new_file_name)
    return handle_crud_result(result)

@dataset_router.delete("/delete-raw-dataset-file")
def delete_raw_dataset_file(
    filename: str = Query(...),
    db: Session = Depends(get_db)
):
    result = delete_raw_dataset(db, filename)
    return handle_crud_result(result)

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
    result = list_datasets(db, skip=skip, limit=limit)
    return handle_crud_result(result)

@dataset_router.get("/dataset-details/{filename}", response_model=dict)
def get_dataset_overview(filename: str, db: Session = Depends(get_db)):
    result = get_dataset_stats(db, filename=filename)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@dataset_router.put("/rename-dataset-file")
def rename_processed_dataset_file(
    dataset_id: int = Query(...),
    new_name: str = Query(...),
    db: Session = Depends(get_db)
):
    result = rename_dataset(db, dataset_id, new_name)
    return handle_crud_result(result)

@dataset_router.delete("/delete-dataset-file")
def delete_processed_dataset_file(
    dataset_id: int = Query(...),
    db: Session = Depends(get_db)
):
    result = delete_dataset(db, dataset_id)
    return handle_crud_result(result)

@dataset_router.post("/preprocess-dataset", status_code=status.HTTP_202_ACCEPTED)
async def preprocess_dataset_endpoint(request: Request):
    data = await request.json()
    executor.submit(
        asyncio.run,
        process_preprocessing(
            data["directory"],
            data["fileName"],
            data["operations"]
        )
    )
    return {"message": "Preprocessing initiated"}



########## Task Management Routes
@dataset_router.post("/tasks", response_model=TaskResponse)
def create_new_task(task: TaskCreate, db: Session = Depends(get_db)):
    result = create_task(db, task)
    return handle_crud_result(result)

@dataset_router.delete("/tasks/{task_id}")
def delete_existing_task(task_id: int, db: Session = Depends(get_db)):
    result = delete_task(db, task_id)
    return handle_crud_result(result)

########### Benchmark Management Routes
@dataset_router.post("/benchmarks", response_model=BenchmarkResponse)
def create_new_benchmark(benchmark: BenchmarkCreate, db: Session = Depends(get_db)):
    result = create_benchmark(db, benchmark)
    return handle_crud_result(result)

@dataset_router.delete("/benchmarks/{benchmark_id}")
def delete_existing_benchmark(benchmark_id: int, db: Session = Depends(get_db)):
    result = delete_benchmark(db, benchmark_id)
    return handle_crud_result(result)

############ HDFS specific routes
@dataset_router.get("/testing_list_all_datasets_from_hdfs")
async def testing_list_all_datasets():
    return await hdfs_client.testing_list_all_datasets()

@dataset_router.get("/list-recent-uploads")   
async def list_recent_uploads():
    return await hdfs_client.list_recent_uploads()


# delete later
# from fastapi import APIRouter, HTTPException, Depends, Query
# from fastapi import Request, Query, HTTPException
# from dotenv import load_dotenv
# from utils.spark_services import SparkSessionManager
# from utils.database_services import DatabaseManager
# from utils.hdfs_services import HDFSServiceManager
# from sqlalchemy.orm import Session
# from utility.db import get_db
# import json
# import asyncio
# import os
# from concurrent.futures import ThreadPoolExecutor

# load_dotenv()
# executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# router = APIRouter(tags=["Preprocessing"])

# HDFS_RAW_DATASETS_DIR = os.getenv("HDFS_RAW_DATASETS_DIR")
# HDFS_PROCESSED_DATASETS_DIR = os.getenv("HDFS_PROCESSED_DATASETS_DIR")
# RECENTLY_UPLOADED_DATASETS_DIR = os.getenv("RECENTLY_UPLOADED_DATASETS_DIR") 

# # Initialize the database
# # db_client = DatabaseManager() #use depends(get_db) instead
# hdfs_client = HDFSServiceManager()
# spark_client = SparkSessionManager()

# async def create_dataset(filename: str, filetype: str):
#     """creates a new dataset in hadoop and adds it to the database table raw_datasets"""
#     try:
#         # rename the hdfs file with suffix __PROCESSING__, not prefix (spark can't read files with starting __)
#         source_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}"
#         destination_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}__PROCESSING__"
#         await hdfs_client.rename_file_or_folder(source_path, destination_path)
#         newfilename = f"{filename}__PROCESSING__"
#         dataset_overview = await spark_client.create_new_dataset(newfilename, filetype)
#         filename = filename.replace("csv", "parquet")
#         await db_client.add_dataset(HDFS_RAW_DATASETS_DIR, filename, dataset_overview)
#         # hdfs_client.delete_file_from_hdfs(RECENTLY_UPLOADED_DATASETS_DIR, newfilename)
#         await hdfs_client.rename_file_or_folder(destination_path,source_path)
#     except Exception as e:
#         source_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}"
#         destination_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}__PROCESSING__"
#         await hdfs_client.rename_file_or_folder(destination_path,source_path)
#         print("Error in processing the data is: ", str(e))


# async def preprocess_dataset(directory: str, filename: str, operations: list):
#     """preprocesses the dataset, saves to new hadoop directory and adds it to the database table datasets"""
#     try:
#         # rename the file with __PROCESSING__ suffix
#         await db_client.rename_dataset(directory,filename, f"{filename}__PROCESSING__")
#         dataset_overview =  await spark_client.preprocess_data(directory, filename, operations)
#         await db_client.rename_dataset(directory, f"{filename}__PROCESSING__",filename)
#         await db_client.add_dataset(HDFS_PROCESSED_DATASETS_DIR, dataset_overview["fileName"], dataset_overview)
#     except Exception as e:
#         await db_client.rename_dataset(directory, f"{filename}__PROCESSING__",filename)
#         # can't delete the file, it will delete prev if not written currently
#         # hdfs_client.delete_file_from_hdfs(HDFS_PROCESSED_DATASETS_DIR, filename)
#         print("Error in preprocessing the data is: ", str(e))

# @router.get("/preprocessing")
# def hello_server():
#     return {"message": "Hello, this is preprocessing router!"}

# @router.get("/testing_list_all_datasets")
# async def testing_list_all_datasets():
#     return await hdfs_client.testing_list_all_datasets()

# @router.get("/list-recent-uploads")   
# async def list_recent_uploads():
#     return await hdfs_client.list_recent_uploads()

# # No such route, like listing all datasets you need to request 2 times, one for raw and one for processed
# # This is because they both may exceed the normal limit so skipping should be done individually

# @router.get("/raw-dataset-overview/{filename}")
# async def get_overview(filename: str, db: Session = Depends(get_db)):
#     return await get_raw_dataset_stats(db, filename)
    
# @router.get("/dataset-overview/{filename}")
# async def get_overview(filename: str, db: Session = Depends(get_db)):
#     return await get_dataset_stats(db, filename)

# @router.get("/list-datasets")
# async def list_all_datasets():
#     return list_datasets(db,skip,limit)

# @router.get("list-raw-datasets")
# # complete


# @router.post("/create-new-dataset")
# async def create_new_dataset(request: Request):
#     data = await request.json()
#     # directory not needed, it is known that directory will be RECENTLY_UPLOADED_DATASETS_DIR from env file
#     filename = data.get("fileName")
#     filetype = filename.split(".")[-1]
#     if filetype not in ["csv", "parquet"]:
#         print("Invalid file type")
#         return {"error": "Invalid file type. Please upload a CSV or Parquet file."}
#     # background_tasks.add_task(create_dataset, filename, filetype)  #this is blocking the server    
#     executor.submit(lambda filename,filetype: asyncio.run(create_dataset(filename,filetype)),filename,filetype)

#     return {"message": "Dataset creation started..."}

# @router.post("/preprocess-dataset")
# async def process_dataset(request: Request):
#     data = await request.json()
#     directory = data.get("directory")
#     filename = data.get("fileName")
#     operations = data.get("operations")
#     print(f"Started processing {filename}...")

#     executor.submit(lambda directory,filename,operations: asyncio.run(preprocess_dataset(directory, filename, operations)), directory,filename,operations)
#     return {"message": "Dataset preprocessed successfully"}

# @router.put("/rename-raw-dataset-file")
# async def rename_file(directory: str = Query(...), oldFileName: str = Query(...), newFileName: str = Query(...)):
#     try:
#         await hdfs_client.rename_file_or_folder(f"{directory}/{oldFileName}", f"{directory}/{newFileName}")
#         await db_client.rename_dataset(directory, oldFileName, newFileName)
#         return {"message": "File renamed successfully!"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.put("/rename-dataset-file")
# # complete

# @router.delete("/delete-raw-dataset-file")
# async def delete_file(directory: str = Query(...), fileName: str = Query(...)):
#     # async because it's not more CPU bound task it's more of I/O bound
#     try:       
#         hdfs_client.delete_file_from_hdfs(directory, fileName)
#         await db_client.delete_dataset(directory, fileName)
#         return {"message": "File deleted successfully!"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

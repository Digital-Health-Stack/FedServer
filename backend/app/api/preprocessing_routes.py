from fastapi import APIRouter, HTTPException
from fastapi import Request, Query, HTTPException
from dotenv import load_dotenv
from utils.spark_services import SparkSessionManager
from utils.database_services import DatabaseManager
from utils.hdfs_services import HDFSServiceManager
import json
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

router = APIRouter(tags=["Preprocessing"])

HDFS_RAW_DATASETS_DIR = os.getenv("HDFS_RAW_DATASETS_DIR")
HDFS_PROCESSED_DATASETS_DIR = os.getenv("HDFS_PROCESSED_DATASETS_DIR")
RECENTLY_UPLOADED_DATASETS_DIR = os.getenv("RECENTLY_UPLOADED_DATASETS_DIR") 

# Initialize the database
db_client = DatabaseManager()
hdfs_client = HDFSServiceManager()
spark_client = SparkSessionManager()

async def create_dataset(filename: str, filetype: str):
    try:
        # rename the hdfs file with suffix __PROCESSING__, not prefix (spark can't read files with starting __)
        source_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}"
        destination_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}__PROCESSING__"
        await hdfs_client.rename_file_or_folder(source_path, destination_path)
        newfilename = f"{filename}__PROCESSING__"
        dataset_overview = await spark_client.create_new_dataset(newfilename, filetype)
        filename = filename.replace("csv", "parquet")
        await db_client.add_dataset(HDFS_RAW_DATASETS_DIR, filename, dataset_overview)
        # hdfs_client.delete_file_from_hdfs(RECENTLY_UPLOADED_DATASETS_DIR, newfilename)
        await hdfs_client.rename_file_or_folder(destination_path,source_path)
    except Exception as e:
        source_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}"
        destination_path = f"{RECENTLY_UPLOADED_DATASETS_DIR}/{filename}__PROCESSING__"
        await hdfs_client.rename_file_or_folder(destination_path,source_path)
        print("Error in processing the data is: ", str(e))


async def preprocess_dataset(directory: str, filename: str, operations: list):
    try:
        # rename the file with __PROCESSING__ suffix
        await db_client.rename_dataset(directory,filename, f"{filename}__PROCESSING__")
        dataset_overview =  await spark_client.preprocess_data(directory, filename, operations)
        await db_client.rename_dataset(directory, f"{filename}__PROCESSING__",filename)
        await db_client.add_dataset(HDFS_PROCESSED_DATASETS_DIR, dataset_overview["fileName"], dataset_overview)
    except Exception as e:
        await db_client.rename_dataset(directory, f"{filename}__PROCESSING__",filename)
        # can't delete the file, it will delete prev if not written currently
        # hdfs_client.delete_file_from_hdfs(HDFS_PROCESSED_DATASETS_DIR, filename)
        print("Error in preprocessing the data is: ", str(e))

@router.get("/preprocessing")
def hello_server():
    return {"message": "Hello, this is preprocessing router!"}

@router.get("/testing_list_all_datasets")
async def testing_list_all_datasets():
    return await hdfs_client.testing_list_all_datasets()

@router.get("/list-recent-uploads")   
async def list_recent_uploads():
    return await hdfs_client.list_recent_uploads()

@router.get("/list-all-datasets")
async def list_all_datasets():
    return await db_client.list_all_datasets()

@router.get("/dataset-overview/{path}/{dataset_id}")
async def get_overview(path: str, dataset_id: str):
    try:
        return await db_client.get_dataset(path, dataset_id)
    except Exception as e:
        return {"error": str(e)}
    
@router.post("/create-new-dataset")
async def create_new_dataset(request: Request):
    data = await request.json()
    # directory not needed, it is known that directory will be RECENTLY_UPLOADED_DATASETS_DIR from env file
    filename = data.get("fileName")
    filetype = filename.split(".")[-1]
    if filetype not in ["csv", "parquet"]:
        print("Invalid file type")
        return {"error": "Invalid file type. Please upload a CSV or Parquet file."}
    # background_tasks.add_task(create_dataset, filename, filetype)  #this is blocking the server    
    executor.submit(lambda filename,filetype: asyncio.run(create_dataset(filename,filetype)),filename,filetype)

    return {"message": "Dataset creation started..."}

@router.post("/preprocess-dataset")
async def process_dataset(request: Request):
    data = await request.json()
    directory = data.get("directory")
    filename = data.get("fileName")
    operations = data.get("operations")
    print(f"Started processing {filename}...")

    executor.submit(lambda directory,filename,operations: asyncio.run(preprocess_dataset(directory, filename, operations)), directory,filename,operations)
    return {"message": "Dataset preprocessed successfully"}

@router.put("/rename-file")
async def rename_file(directory: str = Query(...), oldFileName: str = Query(...), newFileName: str = Query(...)):
    try:
        await hdfs_client.rename_file_or_folder(f"{directory}/{oldFileName}", f"{directory}/{newFileName}")
        await db_client.rename_dataset(directory, oldFileName, newFileName)
        return {"message": "File renamed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/delete-file")
async def delete_file(directory: str = Query(...), fileName: str = Query(...)):
    # async because it's not more CPU bound task it's more of I/O bound
    try:       
        hdfs_client.delete_file_from_hdfs(directory, fileName)
        await db_client.delete_dataset(directory, fileName)
        return {"message": "File deleted successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


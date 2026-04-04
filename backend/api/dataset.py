from fastapi import APIRouter, HTTPException, Depends, Query, status, UploadFile, File, Body
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
import os
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

from schemas.dataset import (
    DatasetCreate,
    Operation,
    DatasetUpdate,
)

from db.datasets_crud import (
    create_dataset,
    delete_dataset,
    rename_dataset,
    list_datasets,
    get_dataset_by_filename,
    get_data_filename_by_id,
    edit_dataset_details,
)

from utilities.core.db import get_db
from services.local_storage import LocalStorageManager, folder_name_from_api_filename
from services.data_processing import DataProcessingManager
from dotenv import load_dotenv

load_dotenv()

executor = ThreadPoolExecutor(max_workers=os.cpu_count())

dataset_router = APIRouter(tags=["Dataset"])

local_client = LocalStorageManager()
data_client = DataProcessingManager()


def _build_dataset_overview(db: Session, filename: str) -> dict:
    meta = local_client.read_metadata(filename)
    if not meta:
        return {"details": "File not found"}
    row = get_dataset_by_filename(db, filename)
    if not row:
        return {"details": "File not found"}
    return {
        "dataset_id": row.dataset_id,
        "filename": meta.get("filename", filename),
        "description": meta.get("description") or row.description,
        "datastats": meta.get("datastats"),
    }


###################### Background processing tasks ######################
async def process_create_dataset(filename: str, filetype: str):
    db = next(get_db())
    print("Processing dataset: ", filename, filetype)
    try:
        desc = f"Dataset created from {filename}"
        dataset_overview = await data_client.create_new_dataset(
            filename, filetype, description=desc
        )
        if "numRows" not in dataset_overview:
            err = dataset_overview.get("message", "Dataset processing failed")
            print("create_new_dataset did not return stats:", dataset_overview)
            return {"error": err}

        print(
            f"Overview of dataset: {dataset_overview['numRows']} rows, {dataset_overview['numColumns']} columns"
        )

        dataset_obj = DatasetCreate(
            filename=dataset_overview["filename"],
            description=dataset_overview.get("description", desc),
        )

        crud_result = create_dataset(db, dataset_obj)
        if isinstance(crud_result, dict) and "error" in crud_result:
            raise HTTPException(status_code=400, detail=crud_result["error"])
        return {"message": "Dataset created successfully"}
    except Exception as e:
        print("Error in processing the data is: ", str(e))
        return {"error": str(e)}
    finally:
        db.close()


async def process_preprocessing(filename: str, operations: List[Operation]):
    stem = folder_name_from_api_filename(filename)
    processing_api = f"{stem}__PROCESSING__.parquet"
    db = next(get_db())
    try:
        await local_client.rename_file_or_folder(filename, processing_api)

        renaming_result = rename_dataset(db, filename, processing_api)
        if isinstance(renaming_result, dict) and "error" in renaming_result:
            raise HTTPException(status_code=400, detail=renaming_result["error"])

        processed_info = await data_client.preprocess_data(
            processing_api,
            [op.model_dump() for op in operations],
        )

        print(
            "Check:: in processing fn- if error goes to except block: ", processed_info
        )
        new_dataset = DatasetCreate(
            filename=processed_info["filename"],
            description=processed_info.get(
                "description", f"Processed version of {filename}"
            ),
        )

        crud_result = create_dataset(db, dataset=new_dataset)
        if isinstance(crud_result, dict) and "error" in crud_result:
            raise HTTPException(status_code=400, detail=crud_result["error"])

        await local_client.rename_file_or_folder(processing_api, filename)

        renaming_result = rename_dataset(db, processing_api, filename)
        if isinstance(renaming_result, dict) and "error" in renaming_result:
            raise HTTPException(status_code=400, detail=renaming_result["error"])
        return {"message": "Preprocessing completed successfully"}

    except Exception as e:
        await local_client.rename_file_or_folder(
            processing_api, filename, ignore_missing=True
        )
        rename_dataset(db, processing_api, filename)
        print("Error in preprocessing the data is: ", str(e))
        return {"error": str(e)}
    finally:
        db.close()


######################## Dataset Routes #######################


@dataset_router.get("/preprocessing", summary="Test server connection")
def hello_server():
    return {"message": "Preprocessing router operational"}


@dataset_router.get("/list-datasets")
def list_datasets_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    try:
        result = list_datasets(db, skip=skip, limit=limit)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        datasets = result.get("datasets", [])
        total = result.get("total", 0)
        return {
            "datasets": [
                {
                    "dataset_id": d.dataset_id,
                    "filename": d.filename,
                    "description": d.description,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in datasets
            ],
            "total": total,
        }
    except HTTPException:
        raise
    except Exception as e:
        print("Error in listing datasets: ", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@dataset_router.get("/dataset-details/{filename}", response_model=dict)
def get_dataset_overview(filename: str, db: Session = Depends(get_db)):
    try:
        result = _build_dataset_overview(db, filename)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        if result.get("details") == "File not found":
            raise HTTPException(status_code=404, detail="Dataset not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        print("Error in getting dataset overview: ", str(e))
        return {"error": str(e)}


@dataset_router.put("/rename-dataset-file")
async def rename_dataset_file(
    dataset_id: int = Query(...),
    new_name: str = Query(...),
    db: Session = Depends(get_db),
):
    try:
        old_file_name = get_data_filename_by_id(db, dataset_id)
        if isinstance(old_file_name, dict) and "error" in old_file_name:
            raise HTTPException(status_code=404, detail=old_file_name["error"])

        await local_client.rename_file_or_folder(old_file_name, new_name)

        result = rename_dataset(db, old_file_name, new_name)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        meta = local_client.read_metadata(new_name) or {}
        meta["filename"] = new_name
        local_client.write_metadata(new_name, meta)
        return result
    except Exception as e:
        print("Error in renaming dataset: ", str(e))
        await local_client.rename_file_or_folder(
            new_name, old_file_name, ignore_missing=True
        )
        return {"error": str(e)}


@dataset_router.put("/edit-dataset-details")
async def edit_dataset_route(newdetails: DatasetUpdate, db: Session = Depends(get_db)):
    try:
        old_file_name = get_data_filename_by_id(db, newdetails.dataset_id)
        if isinstance(old_file_name, dict) and "error" in old_file_name:
            raise HTTPException(status_code=404, detail=old_file_name["error"])

        if old_file_name != newdetails.filename:
            await local_client.rename_file_or_folder(
                old_file_name, newdetails.filename
            )

        result = edit_dataset_details(db, newdetails)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        meta = local_client.read_metadata(newdetails.filename) or {}
        meta["filename"] = newdetails.filename
        if newdetails.description is not None:
            meta["description"] = newdetails.description
        local_client.write_metadata(newdetails.filename, meta)

        return {"message": "Dataset details updated successfully"}
    except Exception as e:
        print("Error in editing dataset details: ", str(e))
        await local_client.rename_file_or_folder(
            newdetails.filename, old_file_name, ignore_missing=True
        )
        return {"error": str(e)}


@dataset_router.delete("/delete-dataset-file")
async def delete_dataset_file(
    dataset_id: int = Query(...), db: Session = Depends(get_db)
):
    try:
        filename = get_data_filename_by_id(db, dataset_id)
        if isinstance(filename, dict) and "error" in filename:
            raise HTTPException(status_code=404, detail=filename["error"])

        await local_client.delete_file(filename)

        result = delete_dataset(db, dataset_id)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        print("Error in deleting dataset: ", str(e))
        return {"error": str(e)}


@dataset_router.post("/create-new-dataset", status_code=status.HTTP_202_ACCEPTED)
async def create_new_dataset(file: UploadFile = File(...)):
    try:
        print(f"Upload started for file: {file.filename}")
        filename = file.filename
        filetype = filename.split(".")[-1].lower()

        if filetype not in ["csv", "parquet"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Supported formats: CSV, Parquet",
            )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            local_client.save_file(temp_file_path, file.filename)

            executor.submit(asyncio.run, process_create_dataset(filename, filetype))

            return JSONResponse(
                status_code=200,
                content={
                    "message": "✅ File saved to server storage; dataset processing started",
                    "filename": file.filename,
                    "storage_path": local_client.get_path(file.filename),
                    "file_size": file.size,
                },
            )

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"❌ Upload failed: {str(e)}")


@dataset_router.post("/process-stored-file", status_code=status.HTTP_202_ACCEPTED)
async def process_stored_file(data: dict = Body(...)):
    """Process an already-uploaded file in storage by filename (used by ViewRecentUploads)."""
    filename = data.get("fileName") or data.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")

    if not local_client.check_file_exists(filename):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in storage")

    filetype = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if filetype not in ["csv", "parquet"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported formats: CSV, Parquet",
        )

    executor.submit(asyncio.run, process_create_dataset(filename, filetype))
    return {
        "message": f"✅ Processing started for '{filename}'",
        "filename": filename,
    }


@dataset_router.post("/preprocess-dataset", status_code=status.HTTP_202_ACCEPTED)
async def preprocess_dataset_endpoint(request: Request):
    data = await request.json()
    executor.submit(
        asyncio.run,
        process_preprocessing(
            data["filename"],
            data["operations"],
        ),
    )
    return {"message": "Preprocessing initiated"}


# Recent Uploads Management
@dataset_router.get("/list-recent-uploads")
async def list_recent_uploads():
    return await local_client.list_recent_uploads()


@dataset_router.delete("/delete-recent-uploaded-file")
async def delete_recent_uploaded_file(
    filename: str = Query(...),
    directory: Optional[str] = Query(None),
):
    _ = directory
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid Delete Request")
    try:
        await local_client.delete_file(filename)
        return {"message": "File deleted successfully"}
    except Exception as e:
        print("Error in deleting recent upload: ", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@dataset_router.get("/list-all-datasets")
def list_all_datasets_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Return all datasets (single table)."""
    try:
        proc_result = list_datasets(db, skip=0, limit=10000)

        combined = []
        if isinstance(proc_result, dict) and "datasets" in proc_result:
            for d in proc_result["datasets"]:
                combined.append({
                    "dataset_id": d.dataset_id,
                    "filename": d.filename,
                    "description": d.description,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                })

        total = len(combined)
        paginated = combined[skip : skip + limit]

        return {
            "datasets": paginated,
            "total": total,
        }
    except Exception as e:
        print("Error in listing all datasets: ", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@dataset_router.get("/dataset-preview/{filename}", response_model=dict)
def get_dataset_preview(
    filename: str,
    n: int = Query(5, ge=0, le=500),
):
    """Return the first N rows of a dataset as JSON records (read from dataset.parquet)."""
    try:
        path = local_client.get_dataset_parquet_path(filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        import pandas as pd
        df = pd.read_parquet(path)
        n = max(0, min(int(n), 500))
        if df.empty or n == 0:
            return {"datasetHead": []}

        rows = df.head(n).to_dict(orient="records")
        import numpy as np

        def _serialize(obj):
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                v = float(obj)
                return v if np.isfinite(v) else None
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            try:
                if obj is not None and pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                pass
            return obj

        return {"datasetHead": _serialize(rows)}
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from services.local_storage import LocalStorageManager
from dotenv import load_dotenv

load_dotenv()

file_upload_router = APIRouter(prefix="/file-upload", tags=["File Upload Router"])

storage_manager = LocalStorageManager()


@file_upload_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"Upload started for file: {file.filename}")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            storage_manager.save_file(temp_file_path, file.filename)

            return JSONResponse(
                status_code=200,
                content={
                    "message": "✅ File uploaded successfully!",
                    "filename": file.filename,
                    "path": storage_manager.get_path(file.filename),
                    "file_size": file.size,
                },
            )

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"❌ Upload failed: {str(e)}")


@file_upload_router.get("/list-files")
async def list_uploaded_files():
    try:
        result = storage_manager.list_files_upload_response()
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        print(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@file_upload_router.delete("/delete/{filename}")
async def delete_uploaded_file(filename: str):
    try:
        await storage_manager.delete_file(filename)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"✅ File {filename} deleted successfully!",
                "filename": filename,
            },
        )

    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"❌ Failed to delete file: {str(e)}"
        )

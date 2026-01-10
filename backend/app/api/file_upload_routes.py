from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from helpers.hdfs_services import HDFSServiceManager
from dotenv import load_dotenv

load_dotenv()

file_upload_router = APIRouter(prefix="/file-upload", tags=["File Upload Router"])

# Initialize HDFS service manager
hdfs_manager = HDFSServiceManager()

# Get HDFS configuration from environment variables
HDFS_URL = os.getenv("HDFS_URL")
HDFS_TARGET_PATH = f"/user/{os.getenv('HADOOP_USER_NAME')}/{os.getenv('RECENTLY_UPLOADED_DATASETS_DIR')}"


@file_upload_router.post("/upload")
async def upload_file_to_hdfs(file: UploadFile = File(...)):
    """
    Upload a file to HDFS using the existing HDFS service infrastructure.

    Args:
        file: The file to upload

    Returns:
        JSON response with upload status
    """
    try:
        print(f"Upload started for file: {file.filename}")

        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp_file:
            # Write the uploaded file content to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            # Use the existing HDFS service to upload the file
            hdfs_path = f"{HDFS_TARGET_PATH}/{file.filename}"

            # Upload using the HDFS client - pass the file path, not the file object
            def upload_to_hdfs(client):
                client.upload(hdfs_path, temp_file_path, overwrite=True)
                print(f"File uploaded to HDFS: {hdfs_path}")
                return {"message": "File uploaded successfully", "hdfs_path": hdfs_path}

            result = hdfs_manager._with_hdfs_client(upload_to_hdfs)

            return JSONResponse(
                status_code=200,
                content={
                    "message": "✅ File uploaded to HDFS successfully!",
                    "filename": file.filename,
                    "hdfs_path": hdfs_path,
                    "file_size": file.size,
                },
            )

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"❌ Upload failed: {str(e)}")


@file_upload_router.get("/list-files")
async def list_uploaded_files():
    """
    List all files in the HDFS upload directory.

    Returns:
        JSON response with list of files
    """
    try:
        # Use the existing list_recent_uploads method but for our target directory
        def list_files(client):
            result = {"contents": {}, "error": None}
            try:
                files = client.list(HDFS_TARGET_PATH, status=True)

                formatted = []
                for entry in files:
                    filename = entry[0]
                    meta = entry[1]

                    if meta["type"] == "FILE":
                        size = meta["length"]
                        formatted.append(
                            {
                                "filename": filename,
                                "size": size,
                                "type": meta["type"],
                                "modification_time": meta.get("modificationTime"),
                                "permission": meta.get("permission"),
                            }
                        )

                result["contents"] = {HDFS_TARGET_PATH: formatted}
                return result

            except Exception as e:
                print(f"Error listing files in HDFS: {e}")
                raise Exception(f"Error listing files in HDFS: {e}")

        result = hdfs_manager._with_hdfs_client(list_files)
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        print(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@file_upload_router.delete("/delete/{filename}")
async def delete_uploaded_file(filename: str):
    """
    Delete a specific file from HDFS.

    Args:
        filename: Name of the file to delete

    Returns:
        JSON response with deletion status
    """
    try:
        # Use the existing delete method
        await hdfs_manager.delete_file_from_hdfs(HDFS_TARGET_PATH, filename)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"✅ File {filename} deleted from HDFS successfully!",
                "filename": filename,
            },
        )

    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"❌ Failed to delete file: {str(e)}"
        )




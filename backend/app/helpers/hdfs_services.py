import os
from hdfs import InsecureClient
from dotenv import load_dotenv

load_dotenv()

HDFS_URL = os.getenv("HDFS_URL")
HADOOP_USER_NAME = os.getenv("HADOOP_USER_NAME")
HDFS_RAW_DATASETS_DIR = os.getenv("HDFS_RAW_DATASETS_DIR")
HDFS_PROCESSED_DATASETS_DIR = os.getenv("HDFS_PROCESSED_DATASETS_DIR")
RECENTLY_UPLOADED_DATASETS_DIR = os.getenv("RECENTLY_UPLOADED_DATASETS_DIR")
"""
NOTE: HDFS session is created and destroyed on demand, so there is no session created when __init__ method is called.
"""
class HDFSServiceManager:
    def __init__(self):
        """
        Initialize HDFSServiceManager with basic settings.
        HDFS connection is not persistent and will be established only when required.
        """
        self.buffer = b""
        self.file_name = ""

    def _with_hdfs_client(self, operation):
        """
        Internal utility to manage HDFS connection asynchronously.
        Offloads HDFS operations to a separate thread to avoid blocking the event loop.
        """
        def wrapped_operation():
            client = InsecureClient(HDFS_URL, user=HADOOP_USER_NAME)
            try:
                return operation(client)
            except Exception as e:
                print(f"Error during HDFS operation: {e}")
                raise Exception(f"Error during HDFS operation: {e}")
            finally:
                client = None  # Explicitly clean up the client

        # Offloading the blocking operation to a thread ...didn't work for some reason
        # return asyncio.to_thread(wrapped_operation)
        return wrapped_operation()

    async def delete_file_from_hdfs(self, directory, filename):
        """
        Delete a file from HDFS, don't make this method async (sync nature required for few use cases)
        """
        hdfs_path = os.path.join(directory, filename)
        print(f"Deleting {hdfs_path} from HDFS...")
        def delete(client):
            status = client.delete(hdfs_path,recursive=True)
            if status:
                print(f"Deleted {hdfs_path} from HDFS.")
            else:
                print(f"Failed to delete {hdfs_path} from HDFS.")
                raise Exception(f"Failed to delete {hdfs_path} from HDFS.")

        try:
            return self._with_hdfs_client(delete)
        except Exception as e:
            raise Exception(f"Error deleting file from HDFS: {e}")

    async def list_recent_uploads(self):
        def human_readable_size(size_in_bytes):
            """Convert size in bytes to human-readable format (KB, MB, GB, etc.)."""
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_in_bytes < 1024.0:
                    return f"{size_in_bytes:.2f} {unit}"
                size_in_bytes /= 1024.0
            return f"{size_in_bytes:.2f} PB"
        
        def get_directory_size(client, path):
            total_size = 0
            try:
                entries = client.list(path, status=True)
                for name, meta in entries:
                    full_path = f"{path}/{name}"
                    if meta['type'] == 'FILE':
                        total_size += meta['length']
                    elif meta['type'] == 'DIRECTORY':
                        total_size += get_directory_size(client, full_path)
            except Exception as e:
                print(f"Error accessing path {path}: {e}")
            return total_size

        def list_files(client):
            result = {'contents': {}, 'error': None}
            try:
                base_path = f"/user/{HADOOP_USER_NAME}/{RECENTLY_UPLOADED_DATASETS_DIR}"
                files = client.list(base_path, status=True)

                formatted = []
                for entry in files:
                    filename = entry[0]
                    meta = entry[1]
                    full_path = f"{base_path}/{filename}"

                    if meta["type"] == "FILE":
                        size = meta["length"]
                    elif meta["type"] == "DIRECTORY":
                        size = get_directory_size(client, full_path)
                    else:
                        continue

                    formatted.append({
                        "filename": filename,
                        "size": human_readable_size(size),
                        "type": meta["type"]
                    })

                result['contents'] = {RECENTLY_UPLOADED_DATASETS_DIR: formatted}
                return result

            except Exception as e:
                print(f"Error listing files in HDFS: {e}")
                raise Exception(f"Error listing files in HDFS: {e}")
        return self._with_hdfs_client(list_files)
    
    async def testing_list_all_datasets(self):
        def list_files(client):
            result = {'contents': {}, 'error': None}
            try:
                files = client.list(f"/user/{HADOOP_USER_NAME}/{RECENTLY_UPLOADED_DATASETS_DIR}", status=True)
                result['contents'] = files
            except Exception as e:
                result['error'] = str(e)
            return result

        return self._with_hdfs_client(list_files)

    async def check_file_exists(self, hdfs_path):
        """
        Check if a file exists in HDFS.
        """
        def check(client):
            try:
                return client.status(hdfs_path, strict=False)
            except Exception as e:
                print(f"Error checking file existence in HDFS: {e}")
                return False

        try:
            return self._with_hdfs_client(check)
        except Exception as e:
            print(f"Error checking file existence in HDFS: {e}")
            raise Exception(f"Error checking file existence in HDFS: {e}")
        
    async def rename_file_or_folder(self,source_path, destination_path):
        """
        Rename a file in HDFS.
        NOTE: If the destination_path already exists and is a directory, the source will be moved into it
        """
        def rename(client):
            client.rename(source_path, destination_path)
            print(f"Renamed {source_path} to {destination_path} in HDFS.")

        try:
            return self._with_hdfs_client(rename)
        except Exception as e:
            print(f"Error renaming file in HDFS: {e}")
            raise Exception(f"Error renaming file in HDFS: {e}")
    def download_folder_from_hdfs(self, hdfs_folder_path, local_destination_path):
        """
        Download a folder from HDFS to local filesystem
        
        Args:
            hdfs_folder_path: Full path to the folder in HDFS
            local_destination_path: Local path where folder should be downloaded
        """
        def download(client):
            if not os.path.exists(local_destination_path):
                os.makedirs(local_destination_path)
                
            # List all files in the HDFS folder
            files = client.list(hdfs_folder_path, status=True)
            
            for file_entry in files:
                hdfs_file_path = os.path.join(hdfs_folder_path, file_entry[0])
                local_file_path = os.path.join(local_destination_path, file_entry[0])
                
                if file_entry[1]["type"] == "FILE":
                    # Download the file
                    client.download(hdfs_file_path, local_file_path)
                    print(f"Downloaded {hdfs_file_path} to {local_file_path}")
                else:
                    # If it's a directory, create it locally and recurse
                    os.makedirs(local_file_path, exist_ok=True)
                    self.download_folder_from_hdfs(hdfs_file_path, local_file_path)
        
        try:
            return self._with_hdfs_client(download)
        except Exception as e:
            print(f"Error downloading folder from HDFS: {e}")
            raise Exception(f"Error downloading folder from HDFS: {e}")

    ########## Don't delete ################
    # this method is never used in the current implementation of FedData

    # def list_all_content(self):
    #     """
    #     List files and directories in the specified HDFS directory.
    #     """
    #     def list_files(client):
    #         result = {'contents': {}, 'error': None}
    #         try:
    #             dirs = client.list(f"/user/{HADOOP_USER_NAME}", status=True)
    #             dirs = [entry[0] for entry in dirs if entry[1]["type"] == "DIRECTORY"] #entry[0] is dir name
    #             print("dirs:", dirs)
    #             for entry in dirs:
    #                 try:
    #                     files = client.list(entry, status=True)
    #                     result['contents'][entry] = [file_name[0] for file_name in files if file_name[1]["type"] == "FILE"]
    #                 except Exception as e:
    #                     result['contents'][entry] = [str(e)]
    #         except Exception as e:
    #             result['error'] = str(e)
    #         return result

    #     return self._with_hdfs_client(list_files)

    # def read_file_from_hdfs(self, hdfs_path):
    #     """
    #     Read a file from HDFS and return its content as a string.
    #     """
    #     def read(client):
    #         with client.read(hdfs_path) as reader:
    #             return reader.read().decode("utf-8")

    #     try:
    #         return self._with_hdfs_client(read)
    #     except Exception as e:
    #         print(f"Error reading file from HDFS: {e}")
    #         return None

    # def download_from_hdfs(self, hdfs_path, local_path):
    #     """
    #     Download a file from HDFS to the local filesystem.
    #     """
    #     def download(client):
    #         client.download(hdfs_path, local_path, overwrite=True)
    #         print(f"Downloaded {hdfs_path} to {local_path}")

    #     try:
    #         return self._with_hdfs_client(download)
    #     except Exception as e:
    #         print(f"Error downloading file from HDFS: {e}")
    #         return None
        
# sample response if list a directory is called without filters on response
# {
#   "contents": [
#     [
#       "Maragakis et al DUDE docking scores and vortex properties.parquet",
#       {
#         "accessTime": 1738518372593,
#         "blockSize": 134217728,
#         "childrenNum": 0,
#         "fileId": 16977,
#         "group": "supergroup",
#         "length": 37598383,
#         "modificationTime": 1738518373840,
#         "owner": "prashu",
#         "pathSuffix": "Maragakis et al DUDE docking scores and vortex properties.parquet",
#         "permission": "644",
#         "replication": 1,
#         "storagePolicy": 0,
#         "type": "FILE"
#       }
#     ],
#     [
#       "health.parquet",
#       {
#         "accessTime": 1738514857243,
#         "blockSize": 134217728,
#         "childrenNum": 0,
#         "fileId": 16609,
#         "group": "supergroup",
#         "length": 3365,
#         "modificationTime": 1736325436275,
#         "owner": "prashu",
#         "pathSuffix": "health.parquet",
#         "permission": "644",
#         "replication": 1,
#         "storagePolicy": 0,
#         "type": "FILE"
#       }
#     ]
#   ],
#   "error": null
# }
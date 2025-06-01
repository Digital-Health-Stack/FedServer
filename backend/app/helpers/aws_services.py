import os
from dotenv import load_dotenv
from typing import List
import boto3
from botocore.exceptions import ClientError

load_dotenv()  # loads AWS_* and BUCKET_NAME from your .env

class S3Services:
    """ If no args are passed, this will use default credentials for QPD bucket"""
    def __init__(self,
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 region_name: str = None,
                 bucket_name: str = None):
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID", "")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self.region_name = region_name or os.getenv("AWS_REGION", "")
        self.bucket_name = bucket_name or os.getenv("BUCKET_NAME", "")

        self.client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )

    def delete_file(self, key: str) -> None:
        """
        Delete a single object from S3.
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            print(f"Deleted '{key}' from '{self.bucket_name}'.")
        except ClientError as e:
            print(f"[ERROR] deleting '{key}': {e}")

    def delete_files(self, keys: List[str]) -> None:
        """
        Delete up to 1,000 keys in a single API call.
        """
        if not keys:
            print("No keys provided; nothing to delete.")
            return

        # Batch in chunks of 1000
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            payload = {"Objects": [{"Key": k} for k in batch], "Quiet": False}
            try:
                resp = self.client.delete_objects(Bucket=self.bucket_name, Delete=payload)
                deleted = [d["Key"] for d in resp.get("Deleted", [])]
                errors  = resp.get("Errors", [])
                if deleted:
                    print("Deleted:", deleted)
                if errors:
                    print("Errors:", errors)
            except ClientError as e:
                print(f"[ERROR] batch delete: {e}")

    def delete_folder(self, prefix: str) -> None:
        """
        Recursively delete all objects under 'prefix/' (skips placeholder keys).
        """
        if not prefix.endswith("/"):
            prefix += "/"

        paginator = self.client.get_paginator("list_objects_v2")
        keys_to_delete: List[str] = []

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):  # skip “folder marker” objects
                    keys_to_delete.append(key)

        if not keys_to_delete:
            print(f"No objects found under '{prefix}'.")
        else:
            self.delete_files(keys_to_delete)

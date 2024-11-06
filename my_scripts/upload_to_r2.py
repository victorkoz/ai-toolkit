import boto3
from botocore.exceptions import ClientError
import os
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
from my_scripts.connect_mongo import get_client
import time
class CloudflareR2Service:
    REGION = "us-east-1"

    def __init__(self,):
      self.cloudflare_r2_model = get_client()["cloudflare_r2"]

    def get_s3_client(self):
        endpoint_url = os.getenv("CLOUDFLARE_R2_ENDPOINT")
        access_key_id = os.getenv("CLOUDFLARE_R2_READ_WRITE_ACCESS_KEY_ID")
        secret_access_key = os.getenv("CLOUDFLARE_R2_READ_WRITE_SECRET_ACCESS_KEY")

        return boto3.client(
            's3',
            region_name=self.REGION,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key
        )

    def store_file(self, bucket_name, key, buffer, content_type):
        s3_client = self.get_s3_client()
        max_attempts = 3  # Number of retry attempts

        for attempt in range(1, max_attempts + 1):
            try:
                # Attempt to upload the file to Cloudflare R2
                response = s3_client.put_object(
                    Bucket=bucket_name,
                    Key=key,
                    Body=buffer,
                    ContentType=content_type
                )

                # Store metadata in MongoDB if upload is successful
                cloudflare_file_info = {
                    "bucketName": bucket_name,
                    "key": key,
                    "contentType": content_type,
                    "uploadedAt": datetime.now(timezone.utc),
                    "_id": ObjectId()
                }
                self.cloudflare_r2_model.insert_one(cloudflare_file_info)

                print(f"File successfully uploaded on attempt {attempt}")
                return response  # Return on successful upload

            except ClientError as e:
                print(f"Attempt {attempt} failed: Error uploading blob to S3:", e)

                # If this was the last attempt, raise the exception
                if attempt == max_attempts:
                    raise e

                # Otherwise, wait 5 seconds before trying again
                print("Retrying in 5 seconds...")
                time.sleep(5)
    def get_file(self, bucket_name, key, download_path):
        s3_client = self.get_s3_client()

        try:
            # Download the file from Cloudflare R2
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            body = response['Body'].read()

            # Write to file if not already present
            if os.path.exists(download_path):
                print(f"File already exists at path: {download_path}")
            else:
                with open(download_path, "wb") as file:
                    file.write(body)

            return response  # Return metadata and ETag info
        except ClientError as e:
            print("Error downloading blob from S3:", e)
            raise e

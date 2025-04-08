from google.cloud import storage
from google.cloud.storage import Blob
from fastapi import UploadFile
import os
from pathlib import Path
import logging
from typing import Optional, List
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GCSHandler:
    def __init__(self, bucket_name: str):
        """Initialize GCS handler with bucket name"""
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    def _get_blob_path(self, *path_parts: str) -> str:
        """Convert path parts to GCS blob path"""
        return '/'.join(path_parts)
    
    async def save_file(self, file: UploadFile, *path_parts: str) -> str:
        """Save a file to GCS"""
        try:
            blob_path = self._get_blob_path(*path_parts)
            print(f"Saving file to GCS path: {blob_path}")  # Debug print
            blob = self.bucket.blob(blob_path)
            
            # Read file content
            content = await file.read()
            
            # Upload to GCS
            blob.upload_from_string(
                content,
                content_type=file.content_type
            )
            
            # Return the GCS path
            return f"gs://{self.bucket.name}/{blob_path}"
            
        except Exception as e:
            logger.error(f"Error saving file to GCS: {str(e)}")
            raise
    
    def save_json(self, data: dict, *path_parts: str) -> str:
        """Save JSON data to GCS"""
        try:
            blob_path = self._get_blob_path(*path_parts)
            blob = self.bucket.blob(blob_path)
            
            # Convert dict to JSON string
            json_str = json.dumps(data, indent=4)
            
            # Upload to GCS
            blob.upload_from_string(
                json_str,
                content_type='application/json'
            )
            
            return f"gs://{self.bucket.name}/{blob_path}"
            
        except Exception as e:
            logger.error(f"Error saving JSON to GCS: {str(e)}")
            raise
    
    def save_plot(self, plot_data: bytes, *path_parts: str) -> str:
        """Save plot image to GCS"""
        try:
            blob_path = self._get_blob_path(*path_parts)
            blob = self.bucket.blob(blob_path)
            
            # Upload to GCS
            blob.upload_from_string(
                plot_data,
                content_type='image/png'
            )
            
            return f"gs://{self.bucket.name}/{blob_path}"
            
        except Exception as e:
            logger.error(f"Error saving plot to GCS: {str(e)}")
            raise
    
    def list_files(self, *path_parts: str) -> List[str]:
        """List files in a GCS directory"""
        try:
            prefix = self._get_blob_path(*path_parts)
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
            
        except Exception as e:
            logger.error(f"Error listing files in GCS: {str(e)}")
            raise
    
    def get_file(self, *path_parts: str) -> bytes:
        """Get file content from GCS"""
        try:
            blob_path = self._get_blob_path(*path_parts)
            blob = self.bucket.blob(blob_path)
            return blob.download_as_bytes()
            
        except Exception as e:
            logger.error(f"Error getting file from GCS: {str(e)}")
            raise
    
    def get_json(self, *path_parts: str) -> dict:
        """Get JSON data from GCS"""
        try:
            content = self.get_file(*path_parts)
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error getting JSON from GCS: {str(e)}")
            raise
    
    def delete_file(self, *path_parts: str) -> None:
        """Delete file from GCS"""
        try:
            blob_path = self._get_blob_path(*path_parts)
            blob = self.bucket.blob(blob_path)
            blob.delete()
            
        except Exception as e:
            logger.error(f"Error deleting file from GCS: {str(e)}")
            raise
    
    def get_signed_url(self, *path_parts: str, expiration: int = 3600) -> str:
        """Get signed URL for temporary access to file"""
        try:
            blob_path = self._get_blob_path(*path_parts)
            blob = self.bucket.blob(blob_path)
            return blob.generate_signed_url(
                version="v4",
                expiration=datetime.utcnow() + timedelta(seconds=expiration),
                method="GET"
            )
            
        except Exception as e:
            logger.error(f"Error generating signed URL: {str(e)}")
            raise 
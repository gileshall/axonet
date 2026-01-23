"""Google Cloud Storage backend."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from ..base import StorageBackend


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""
    
    def __init__(
        self,
        project: str,
        bucket: str,
        credentials_path: Optional[str] = None,
    ):
        self.project = project
        self.bucket = bucket
        self.credentials_path = credentials_path
        self._client = None
        self._bucket_obj = None
    
    @property
    def client(self):
        if self._client is None:
            from google.cloud import storage
            if self.credentials_path:
                self._client = storage.Client.from_service_account_json(
                    self.credentials_path
                )
            else:
                self._client = storage.Client(project=self.project)
        return self._client
    
    @property
    def bucket_obj(self):
        if self._bucket_obj is None:
            self._bucket_obj = self.client.bucket(self.bucket)
        return self._bucket_obj
    
    def _parse_gcs_path(self, path: str) -> str:
        """Convert gs://bucket/path or path to just the blob path."""
        if path.startswith("gs://"):
            parsed = urlparse(path)
            return parsed.path.lstrip("/")
        return path.lstrip("/")
    
    def _to_gcs_uri(self, blob_path: str) -> str:
        return f"gs://{self.bucket}/{blob_path}"
    
    def upload(self, local_path: Path, remote_path: str) -> str:
        blob_path = self._parse_gcs_path(remote_path)
        blob = self.bucket_obj.blob(blob_path)
        blob.upload_from_filename(str(local_path))
        return self._to_gcs_uri(blob_path)
    
    def download(self, remote_path: str, local_path: Path) -> Path:
        blob_path = self._parse_gcs_path(remote_path)
        blob = self.bucket_obj.blob(blob_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        return local_path
    
    def list_files(self, remote_prefix: str) -> List[str]:
        prefix = self._parse_gcs_path(remote_prefix)
        blobs = self.client.list_blobs(self.bucket, prefix=prefix)
        return [self._to_gcs_uri(blob.name) for blob in blobs]
    
    def exists(self, remote_path: str) -> bool:
        blob_path = self._parse_gcs_path(remote_path)
        blob = self.bucket_obj.blob(blob_path)
        return blob.exists()
    
    def delete(self, remote_path: str) -> None:
        blob_path = self._parse_gcs_path(remote_path)
        
        if blob_path.endswith("/"):
            blobs = list(self.client.list_blobs(self.bucket, prefix=blob_path))
            for blob in blobs:
                blob.delete()
        else:
            blob = self.bucket_obj.blob(blob_path)
            if blob.exists():
                blob.delete()
    
    def get_signed_url(self, remote_path: str, expiration_minutes: int = 60) -> str:
        """Get a signed URL for temporary access."""
        import datetime
        blob_path = self._parse_gcs_path(remote_path)
        blob = self.bucket_obj.blob(blob_path)
        return blob.generate_signed_url(
            expiration=datetime.timedelta(minutes=expiration_minutes),
            method="GET",
        )

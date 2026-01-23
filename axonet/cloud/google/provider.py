"""Google Cloud provider combining storage, compute, and batch."""

from __future__ import annotations

import os
from typing import Optional

from ..base import BatchBackend, CloudProvider, ComputeBackend, StorageBackend
from .batch import GoogleBatch
from .compute import GCECompute
from .storage import GCSStorage


class GoogleCloudProvider(CloudProvider):
    """Google Cloud provider for axonet workloads."""
    
    def __init__(self):
        self._storage: Optional[GCSStorage] = None
        self._compute: Optional[GCECompute] = None
        self._batch: Optional[GoogleBatch] = None
        self._configured = False
    
    @property
    def name(self) -> str:
        return "google"
    
    @property
    def storage(self) -> StorageBackend:
        if self._storage is None:
            raise RuntimeError("Provider not configured. Call configure() first.")
        return self._storage
    
    @property
    def compute(self) -> ComputeBackend:
        if self._compute is None:
            raise RuntimeError("Provider not configured. Call configure() first.")
        return self._compute
    
    @property
    def batch(self) -> BatchBackend:
        if self._batch is None:
            raise RuntimeError("Provider not configured. Call configure() first.")
        return self._batch
    
    def configure(
        self,
        project: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        bucket: Optional[str] = None,
        credentials_path: Optional[str] = None,
        service_account: Optional[str] = None,
        network: str = "default",
        subnetwork: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Configure Google Cloud provider.
        
        Args:
            project: GCP project ID (or GOOGLE_CLOUD_PROJECT env var)
            region: GCP region for Batch (or GOOGLE_CLOUD_REGION env var)
            zone: GCP zone for Compute (or GOOGLE_CLOUD_ZONE env var)
            bucket: GCS bucket name (or AXONET_GCS_BUCKET env var)
            credentials_path: Path to service account JSON (or GOOGLE_APPLICATION_CREDENTIALS)
            service_account: Service account email for workloads. If None, jobs use
                the project's default compute service account. For local API calls,
                uses Application Default Credentials (run `gcloud auth application-default login`).
            network: VPC network name
            subnetwork: VPC subnetwork name
        """
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        region = region or os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        zone = zone or os.environ.get("GOOGLE_CLOUD_ZONE", f"{region}-a")
        bucket = bucket or os.environ.get("AXONET_GCS_BUCKET")
        credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        service_account = service_account or os.environ.get("AXONET_SERVICE_ACCOUNT")
        
        if not project:
            raise ValueError(
                "GCP project required. Set GOOGLE_CLOUD_PROJECT env var or pass project="
            )
        if not bucket:
            raise ValueError(
                "GCS bucket required. Set AXONET_GCS_BUCKET env var or pass bucket="
            )
        
        self._storage = GCSStorage(
            project=project,
            bucket=bucket,
            credentials_path=credentials_path,
        )
        
        self._compute = GCECompute(
            project=project,
            zone=zone,
            bucket=bucket,
            credentials_path=credentials_path,
            network=network,
            subnetwork=subnetwork,
            service_account=service_account,
        )
        
        self._batch = GoogleBatch(
            project=project,
            region=region,
            bucket=bucket,
            credentials_path=credentials_path,
            service_account=service_account,
            network=network,
            subnetwork=subnetwork,
        )
        
        self._configured = True

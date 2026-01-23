"""Abstract base classes for cloud infrastructure."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class JobConfig:
    """Configuration for a compute job."""
    name: str
    command: List[str]
    image: str = "axonet:latest"
    cpu: int = 4
    memory_gb: int = 16
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    disk_gb: int = 100
    max_runtime_hours: float = 24.0
    env: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    input_paths: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    
    retries: int = 0
    priority: int = 0
    
    # Explicit machine type (e.g., "n1-standard-8", "n1-highmem-4")
    # If None, auto-selected based on cpu/memory
    machine_type: Optional[str] = None


@dataclass
class JobResult:
    """Result of a completed job."""
    job_id: str
    status: JobStatus
    exit_code: Optional[int] = None
    logs: Optional[str] = None
    output_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class StorageBackend(ABC):
    """Abstract storage backend for data I/O."""
    
    @abstractmethod
    def upload(self, local_path: Path, remote_path: str) -> str:
        """Upload file to remote storage. Returns remote URI."""
        ...
    
    @abstractmethod
    def download(self, remote_path: str, local_path: Path) -> Path:
        """Download file from remote storage. Returns local path."""
        ...
    
    @abstractmethod
    def list_files(self, remote_prefix: str) -> List[str]:
        """List files under a remote prefix."""
        ...
    
    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if remote path exists."""
        ...
    
    @abstractmethod
    def delete(self, remote_path: str) -> None:
        """Delete remote file or directory."""
        ...
    
    def sync_up(self, local_dir: Path, remote_prefix: str) -> List[str]:
        """Sync local directory to remote. Returns list of uploaded URIs."""
        uploaded = []
        for path in local_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(local_dir)
                remote = f"{remote_prefix.rstrip('/')}/{rel}"
                uploaded.append(self.upload(path, remote))
        return uploaded
    
    def sync_down(self, remote_prefix: str, local_dir: Path) -> List[Path]:
        """Sync remote prefix to local directory. Returns list of downloaded paths."""
        downloaded = []
        for remote in self.list_files(remote_prefix):
            rel = remote[len(remote_prefix):].lstrip("/")
            local = local_dir / rel
            downloaded.append(self.download(remote, local))
        return downloaded


class ComputeBackend(ABC):
    """Abstract compute backend for running jobs."""
    
    @abstractmethod
    def submit(self, config: JobConfig) -> str:
        """Submit a job. Returns job ID."""
        ...
    
    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Get job status."""
        ...
    
    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel a running job."""
        ...
    
    @abstractmethod
    def logs(self, job_id: str) -> str:
        """Get job logs."""
        ...
    
    @abstractmethod
    def wait(self, job_id: str, timeout_seconds: Optional[float] = None) -> JobResult:
        """Wait for job to complete. Returns result."""
        ...
    
    @abstractmethod
    def list_jobs(self, labels: Optional[Dict[str, str]] = None) -> List[str]:
        """List job IDs, optionally filtered by labels."""
        ...


class BatchBackend(ABC):
    """Abstract batch processing backend for parallel jobs."""
    
    @abstractmethod
    def submit_array(self, configs: List[JobConfig], parallelism: int = 10) -> str:
        """Submit array of jobs. Returns batch ID."""
        ...
    
    @abstractmethod
    def status(self, batch_id: str) -> Dict[str, JobStatus]:
        """Get status of all jobs in batch. Returns {task_id: status}."""
        ...
    
    @abstractmethod
    def cancel(self, batch_id: str) -> None:
        """Cancel all jobs in batch."""
        ...
    
    @abstractmethod
    def wait(self, batch_id: str, timeout_seconds: Optional[float] = None) -> Dict[str, JobResult]:
        """Wait for all jobs to complete. Returns {task_id: result}."""
        ...


class CloudProvider(ABC):
    """Abstract cloud provider combining storage, compute, and batch."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'google', 'local')."""
        ...
    
    @property
    @abstractmethod
    def storage(self) -> StorageBackend:
        """Get storage backend."""
        ...
    
    @property
    @abstractmethod
    def compute(self) -> ComputeBackend:
        """Get compute backend."""
        ...
    
    @property
    @abstractmethod
    def batch(self) -> BatchBackend:
        """Get batch backend."""
        ...
    
    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Configure provider with credentials/settings."""
        ...

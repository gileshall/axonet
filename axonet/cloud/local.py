"""Local execution backend for development and testing."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .base import (
    BatchBackend,
    CloudProvider,
    ComputeBackend,
    JobConfig,
    JobResult,
    JobStatus,
    StorageBackend,
)


@dataclass
class LocalJobState:
    config: JobConfig
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    logs: str = ""
    exit_code: Optional[int] = None


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _resolve(self, remote_path: str) -> Path:
        if remote_path.startswith("file://"):
            remote_path = remote_path[7:]
        return self.base_dir / remote_path.lstrip("/")
    
    def upload(self, local_path: Path, remote_path: str) -> str:
        dest = self._resolve(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return f"file://{dest}"
    
    def download(self, remote_path: str, local_path: Path) -> Path:
        src = self._resolve(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return local_path
    
    def list_files(self, remote_prefix: str) -> List[str]:
        prefix_path = self._resolve(remote_prefix)
        if not prefix_path.exists():
            return []
        if prefix_path.is_file():
            return [f"file://{prefix_path}"]
        return [f"file://{p}" for p in prefix_path.rglob("*") if p.is_file()]
    
    def exists(self, remote_path: str) -> bool:
        return self._resolve(remote_path).exists()
    
    def delete(self, remote_path: str) -> None:
        path = self._resolve(remote_path)
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


class LocalCompute(ComputeBackend):
    """Local subprocess-based compute backend."""
    
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, LocalJobState] = {}
    
    def submit(self, config: JobConfig) -> str:
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        env.update(config.env)
        
        log_file = job_dir / "output.log"
        
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                config.command,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(job_dir),
            )
        
        state = LocalJobState(
            config=config,
            process=process,
            start_time=time.time(),
        )
        self._jobs[job_id] = state
        
        return job_id
    
    def status(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            return JobStatus.UNKNOWN
        
        state = self._jobs[job_id]
        if state.process is None:
            return JobStatus.PENDING
        
        poll = state.process.poll()
        if poll is None:
            return JobStatus.RUNNING
        
        state.exit_code = poll
        state.end_time = time.time()
        
        return JobStatus.SUCCEEDED if poll == 0 else JobStatus.FAILED
    
    def cancel(self, job_id: str) -> None:
        if job_id not in self._jobs:
            return
        state = self._jobs[job_id]
        if state.process and state.process.poll() is None:
            state.process.terminate()
            state.process.wait(timeout=10)
    
    def logs(self, job_id: str) -> str:
        job_dir = self.work_dir / job_id
        log_file = job_dir / "output.log"
        if log_file.exists():
            return log_file.read_text()
        return ""
    
    def wait(self, job_id: str, timeout_seconds: Optional[float] = None) -> JobResult:
        if job_id not in self._jobs:
            return JobResult(job_id=job_id, status=JobStatus.UNKNOWN)
        
        state = self._jobs[job_id]
        if state.process:
            state.process.wait(timeout=timeout_seconds)
        
        status = self.status(job_id)
        duration = None
        if state.start_time and state.end_time:
            duration = state.end_time - state.start_time
        
        return JobResult(
            job_id=job_id,
            status=status,
            exit_code=state.exit_code,
            logs=self.logs(job_id),
            duration_seconds=duration,
        )
    
    def list_jobs(self, labels: Optional[Dict[str, str]] = None) -> List[str]:
        if labels is None:
            return list(self._jobs.keys())
        return [
            jid for jid, state in self._jobs.items()
            if all(state.config.labels.get(k) == v for k, v in labels.items())
        ]


class LocalBatch(BatchBackend):
    """Local thread-pool based batch backend."""
    
    def __init__(self, compute: LocalCompute, max_workers: int = 4):
        self.compute = compute
        self.max_workers = max_workers
        self._batches: Dict[str, List[str]] = {}
    
    def submit_array(self, configs: List[JobConfig], parallelism: int = 10) -> str:
        batch_id = f"batch-{uuid.uuid4().hex[:8]}"
        
        job_ids = []
        workers = min(parallelism, self.max_workers, len(configs))
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.compute.submit, cfg): i for i, cfg in enumerate(configs)}
            for future in as_completed(futures):
                job_ids.append(future.result())
        
        self._batches[batch_id] = job_ids
        return batch_id
    
    def status(self, batch_id: str) -> Dict[str, JobStatus]:
        if batch_id not in self._batches:
            return {}
        return {jid: self.compute.status(jid) for jid in self._batches[batch_id]}
    
    def cancel(self, batch_id: str) -> None:
        if batch_id not in self._batches:
            return
        for jid in self._batches[batch_id]:
            self.compute.cancel(jid)
    
    def wait(self, batch_id: str, timeout_seconds: Optional[float] = None) -> Dict[str, JobResult]:
        if batch_id not in self._batches:
            return {}
        
        results = {}
        for jid in self._batches[batch_id]:
            results[jid] = self.compute.wait(jid, timeout_seconds)
        return results


class LocalProvider(CloudProvider):
    """Local execution provider for development and testing."""
    
    def __init__(self):
        self._base_dir: Optional[Path] = None
        self._storage: Optional[LocalStorage] = None
        self._compute: Optional[LocalCompute] = None
        self._batch: Optional[LocalBatch] = None
    
    @property
    def name(self) -> str:
        return "local"
    
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
        base_dir: Optional[str] = None,
        max_workers: int = 4,
        **kwargs,
    ) -> None:
        if base_dir is None:
            base_dir = os.environ.get("AXONET_LOCAL_DIR", "/tmp/axonet")
        
        self._base_dir = Path(base_dir)
        self._storage = LocalStorage(self._base_dir / "storage")
        self._compute = LocalCompute(self._base_dir / "jobs")
        self._batch = LocalBatch(self._compute, max_workers=max_workers)

"""Google Batch backend for parallel dataset generation."""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from ..base import BatchBackend, JobConfig, JobResult, JobStatus


class GoogleBatch(BatchBackend):
    """Google Batch backend for parallel processing (dataset generation)."""
    
    def __init__(
        self,
        project: str,
        region: str,
        bucket: str,
        credentials_path: Optional[str] = None,
        service_account: Optional[str] = None,
        network: str = "default",
        subnetwork: Optional[str] = None,
    ):
        self.project = project
        self.region = region
        self.bucket = bucket
        self.credentials_path = credentials_path
        self.service_account = service_account
        self.network = network
        self.subnetwork = subnetwork
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from google.cloud import batch_v1
            self._client = batch_v1.BatchServiceClient()
        return self._client
    
    def _config_to_task_spec(self, config: JobConfig, task_index: int) -> dict:
        """Convert JobConfig to Batch task spec."""
        env_vars = [
            {"name": k, "value": v} for k, v in config.env.items()
        ]
        env_vars.append({"name": "BATCH_TASK_INDEX", "value": str(task_index)})
        
        return {
            "runnables": [
                {
                    "container": {
                        "image_uri": config.image,
                        "commands": config.command,
                    },
                    "environment": {"variables": {e["name"]: e["value"] for e in env_vars}},
                }
            ],
            "compute_resource": {
                "cpu_milli": config.cpu * 1000,
                "memory_mib": config.memory_gb * 1024,
            },
            "max_retry_count": config.retries,
            "max_run_duration": f"{int(config.max_runtime_hours * 3600)}s",
        }
    
    def submit_array(self, configs: List[JobConfig], parallelism: int = 10) -> str:
        from google.cloud import batch_v1
        
        if not configs:
            raise ValueError("No job configs provided")
        
        batch_id = f"axonet-batch-{uuid.uuid4().hex[:8]}"
        base_config = configs[0]
        
        job = batch_v1.Job()
        job.name = f"projects/{self.project}/locations/{self.region}/jobs/{batch_id}"
        
        task_group = batch_v1.TaskGroup()
        task_group.task_count = len(configs)
        task_group.parallelism = min(parallelism, len(configs))
        
        task_spec = batch_v1.TaskSpec()
        
        runnable = batch_v1.Runnable()
        runnable.container = batch_v1.Runnable.Container()
        runnable.container.image_uri = base_config.image
        runnable.container.commands = base_config.command
        
        runnable.environment = batch_v1.Environment()
        runnable.environment.variables = base_config.env
        
        task_spec.runnables = [runnable]
        
        task_spec.compute_resource = batch_v1.ComputeResource()
        task_spec.compute_resource.cpu_milli = base_config.cpu * 1000
        task_spec.compute_resource.memory_mib = base_config.memory_gb * 1024
        
        task_spec.max_retry_count = base_config.retries
        task_spec.max_run_duration = f"{int(base_config.max_runtime_hours * 3600)}s"
        
        task_group.task_spec = task_spec
        job.task_groups = [task_group]
        
        job.allocation_policy = batch_v1.AllocationPolicy()
        
        instance_policy = batch_v1.AllocationPolicy.InstancePolicy()
        
        # Set explicit machine type if specified
        if base_config.machine_type:
            instance_policy.machine_type = base_config.machine_type
        
        if base_config.gpu_count > 0 and base_config.gpu_type:
            gpu_map = {
                "t4": "nvidia-tesla-t4",
                "v100": "nvidia-tesla-v100", 
                "a100": "nvidia-tesla-a100",
                "l4": "nvidia-l4",
            }
            gpu_type = gpu_map.get(base_config.gpu_type.lower(), base_config.gpu_type)
            
            accelerator = batch_v1.AllocationPolicy.Accelerator()
            accelerator.type_ = gpu_type
            accelerator.count = base_config.gpu_count
            instance_policy.accelerators = [accelerator]
        
        instance_policy_or_template = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
        instance_policy_or_template.policy = instance_policy
        job.allocation_policy.instances = [instance_policy_or_template]
        
        network_interface = batch_v1.AllocationPolicy.NetworkInterface()
        network_interface.network = f"projects/{self.project}/global/networks/{self.network}"
        if self.subnetwork:
            network_interface.subnetwork = f"projects/{self.project}/regions/{self.region}/subnetworks/{self.subnetwork}"
        
        network_policy = batch_v1.AllocationPolicy.NetworkPolicy()
        network_policy.network_interfaces = [network_interface]
        job.allocation_policy.network = network_policy
        
        if self.service_account:
            service_account = batch_v1.ServiceAccount()
            service_account.email = self.service_account
            job.allocation_policy.service_account = service_account
        
        job.logs_policy = batch_v1.LogsPolicy()
        job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        
        job.labels = {
            "axonet-batch": "true",
            "batch-name": base_config.name,
            **{k.replace("_", "-"): v for k, v in base_config.labels.items()},
        }
        
        request = batch_v1.CreateJobRequest()
        request.parent = f"projects/{self.project}/locations/{self.region}"
        request.job_id = batch_id
        request.job = job
        
        created_job = self.client.create_job(request=request)
        
        return batch_id
    
    def _get_job(self, batch_id: str):
        from google.cloud import batch_v1
        
        request = batch_v1.GetJobRequest()
        request.name = f"projects/{self.project}/locations/{self.region}/jobs/{batch_id}"
        return self.client.get_job(request=request)
    
    def _batch_state_to_status(self, state) -> JobStatus:
        from google.cloud import batch_v1
        
        state_map = {
            batch_v1.JobStatus.State.STATE_UNSPECIFIED: JobStatus.UNKNOWN,
            batch_v1.JobStatus.State.QUEUED: JobStatus.PENDING,
            batch_v1.JobStatus.State.SCHEDULED: JobStatus.PENDING,
            batch_v1.JobStatus.State.RUNNING: JobStatus.RUNNING,
            batch_v1.JobStatus.State.SUCCEEDED: JobStatus.SUCCEEDED,
            batch_v1.JobStatus.State.FAILED: JobStatus.FAILED,
            batch_v1.JobStatus.State.DELETION_IN_PROGRESS: JobStatus.CANCELLED,
        }
        return state_map.get(state, JobStatus.UNKNOWN)
    
    def status(self, batch_id: str) -> Dict[str, JobStatus]:
        job = self._get_job(batch_id)
        
        overall_status = self._batch_state_to_status(job.status.state)
        
        task_count = job.task_groups[0].task_count if job.task_groups else 0
        
        results = {}
        for i in range(task_count):
            results[f"task-{i}"] = overall_status
        
        return results
    
    def cancel(self, batch_id: str) -> None:
        from google.cloud import batch_v1
        
        request = batch_v1.DeleteJobRequest()
        request.name = f"projects/{self.project}/locations/{self.region}/jobs/{batch_id}"
        
        self.client.delete_job(request=request)
    
    def wait(self, batch_id: str, timeout_seconds: Optional[float] = None) -> Dict[str, JobResult]:
        start = time.time()
        
        while True:
            job = self._get_job(batch_id)
            status = self._batch_state_to_status(job.status.state)
            
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                break
            
            if timeout_seconds and (time.time() - start) > timeout_seconds:
                break
            
            time.sleep(30)
        
        task_count = job.task_groups[0].task_count if job.task_groups else 0
        
        results = {}
        for i in range(task_count):
            results[f"task-{i}"] = JobResult(
                job_id=f"{batch_id}/task-{i}",
                status=status,
                duration_seconds=time.time() - start,
            )
        
        return results
    
    def get_logs_url(self, batch_id: str) -> str:
        """Get URL to view logs in Cloud Console."""
        return (
            f"https://console.cloud.google.com/batch/jobsDetail/"
            f"locations/{self.region}/jobs/{batch_id}?project={self.project}"
        )

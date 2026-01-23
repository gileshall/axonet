"""Google Compute Engine backend for training jobs."""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from ..base import ComputeBackend, JobConfig, JobResult, JobStatus


class GCECompute(ComputeBackend):
    """Google Compute Engine backend for single-instance jobs (training)."""
    
    def __init__(
        self,
        project: str,
        zone: str,
        bucket: str,
        credentials_path: Optional[str] = None,
        network: str = "default",
        subnetwork: Optional[str] = None,
        service_account: Optional[str] = None,
    ):
        self.project = project
        self.zone = zone
        self.bucket = bucket
        self.credentials_path = credentials_path
        self.network = network
        self.subnetwork = subnetwork
        self.service_account = service_account
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from google.cloud import compute_v1
            self._client = compute_v1.InstancesClient()
        return self._client
    
    def _get_machine_type(self, config: JobConfig) -> str:
        """Map job config to GCE machine type."""
        # Explicit machine type takes precedence
        if config.machine_type:
            return config.machine_type
        
        if config.gpu_count > 0:
            return f"n1-standard-{config.cpu}"
        
        if config.memory_gb <= 8:
            return f"e2-standard-{max(2, config.cpu)}"
        elif config.memory_gb <= 32:
            return f"n2-standard-{max(4, config.cpu)}"
        else:
            return f"n2-highmem-{max(8, config.cpu)}"
    
    def _get_accelerator_config(self, config: JobConfig) -> Optional[Dict]:
        """Get GPU accelerator config if needed."""
        if config.gpu_count == 0 or not config.gpu_type:
            return None
        
        gpu_map = {
            "t4": "nvidia-tesla-t4",
            "v100": "nvidia-tesla-v100",
            "a100": "nvidia-tesla-a100",
            "l4": "nvidia-l4",
        }
        gpu_type = gpu_map.get(config.gpu_type.lower(), config.gpu_type)
        
        return {
            "accelerator_type": f"zones/{self.zone}/acceleratorTypes/{gpu_type}",
            "accelerator_count": config.gpu_count,
        }
    
    def _build_startup_script(self, config: JobConfig) -> str:
        """Build startup script for the instance."""
        env_exports = "\n".join(f'export {k}="{v}"' for k, v in config.env.items())
        command = " ".join(config.command)
        
        script = f"""#!/bin/bash
set -e

{env_exports}

echo "Starting job: {config.name}"
echo "Command: {command}"

{command}

EXIT_CODE=$?
echo "Job completed with exit code: $EXIT_CODE"

gsutil cp /var/log/syslog gs://{self.bucket}/jobs/{config.name}/syslog.txt || true

if [ $EXIT_CODE -eq 0 ]; then
    gcloud compute instances add-metadata $(hostname) --zone={self.zone} --metadata=job-status=succeeded
else
    gcloud compute instances add-metadata $(hostname) --zone={self.zone} --metadata=job-status=failed
fi

shutdown -h now
"""
        return script
    
    def submit(self, config: JobConfig) -> str:
        from google.cloud import compute_v1
        
        job_id = f"axonet-{config.name}-{uuid.uuid4().hex[:6]}"
        
        machine_type = self._get_machine_type(config)
        
        instance = compute_v1.Instance()
        instance.name = job_id
        instance.machine_type = f"zones/{self.zone}/machineTypes/{machine_type}"
        
        instance.labels = {
            "axonet-job": "true",
            "job-name": config.name,
            **{k.replace("_", "-"): v for k, v in config.labels.items()},
        }
        
        instance.metadata = compute_v1.Metadata()
        instance.metadata.items = [
            compute_v1.Items(key="startup-script", value=self._build_startup_script(config)),
            compute_v1.Items(key="job-status", value="running"),
        ]
        
        disk = compute_v1.AttachedDisk()
        disk.auto_delete = True
        disk.boot = True
        init_params = compute_v1.AttachedDiskInitializeParams()
        
        if config.gpu_count > 0:
            init_params.source_image = "projects/ml-images/global/images/family/common-cu121-debian-11"
        else:
            init_params.source_image = "projects/debian-cloud/global/images/family/debian-11"
        
        init_params.disk_size_gb = config.disk_gb
        disk.initialize_params = init_params
        instance.disks = [disk]
        
        accelerator = self._get_accelerator_config(config)
        if accelerator:
            guest_accel = compute_v1.AcceleratorConfig()
            guest_accel.accelerator_type = accelerator["accelerator_type"]
            guest_accel.accelerator_count = accelerator["accelerator_count"]
            instance.guest_accelerators = [guest_accel]
            
            instance.scheduling = compute_v1.Scheduling()
            instance.scheduling.on_host_maintenance = "TERMINATE"
        
        network_interface = compute_v1.NetworkInterface()
        network_interface.network = f"global/networks/{self.network}"
        if self.subnetwork:
            network_interface.subnetwork = f"regions/{self.zone.rsplit('-', 1)[0]}/subnetworks/{self.subnetwork}"
        
        access_config = compute_v1.AccessConfig()
        access_config.name = "External NAT"
        access_config.type_ = "ONE_TO_ONE_NAT"
        network_interface.access_configs = [access_config]
        instance.network_interfaces = [network_interface]
        
        if self.service_account:
            sa = compute_v1.ServiceAccount()
            sa.email = self.service_account
            sa.scopes = [
                "https://www.googleapis.com/auth/cloud-platform",
            ]
            instance.service_accounts = [sa]
        
        request = compute_v1.InsertInstanceRequest()
        request.project = self.project
        request.zone = self.zone
        request.instance_resource = instance
        
        operation = self.client.insert(request=request)
        operation.result()
        
        return job_id
    
    def status(self, job_id: str) -> JobStatus:
        from google.cloud import compute_v1
        from google.api_core.exceptions import NotFound
        
        request = compute_v1.GetInstanceRequest()
        request.project = self.project
        request.zone = self.zone
        request.instance = job_id
        
        try:
            instance = self.client.get(request=request)
        except NotFound:
            return JobStatus.UNKNOWN
        
        if instance.status == "TERMINATED":
            for item in instance.metadata.items:
                if item.key == "job-status":
                    if item.value == "succeeded":
                        return JobStatus.SUCCEEDED
                    elif item.value == "failed":
                        return JobStatus.FAILED
            return JobStatus.FAILED
        
        if instance.status in ("PROVISIONING", "STAGING"):
            return JobStatus.PENDING
        
        if instance.status == "RUNNING":
            return JobStatus.RUNNING
        
        return JobStatus.UNKNOWN
    
    def cancel(self, job_id: str) -> None:
        from google.cloud import compute_v1
        
        request = compute_v1.DeleteInstanceRequest()
        request.project = self.project
        request.zone = self.zone
        request.instance = job_id
        
        operation = self.client.delete(request=request)
        operation.result()
    
    def logs(self, job_id: str) -> str:
        from google.cloud import storage
        
        client = storage.Client(project=self.project)
        bucket = client.bucket(self.bucket)
        
        log_paths = [
            f"jobs/{job_id}/output.log",
            f"jobs/{job_id}/syslog.txt",
        ]
        
        logs = []
        for path in log_paths:
            blob = bucket.blob(path)
            if blob.exists():
                logs.append(blob.download_as_text())
        
        return "\n".join(logs)
    
    def wait(self, job_id: str, timeout_seconds: Optional[float] = None) -> JobResult:
        start = time.time()
        
        while True:
            status = self.status(job_id)
            
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                break
            
            if timeout_seconds and (time.time() - start) > timeout_seconds:
                break
            
            time.sleep(30)
        
        return JobResult(
            job_id=job_id,
            status=status,
            logs=self.logs(job_id),
            duration_seconds=time.time() - start,
        )
    
    def list_jobs(self, labels: Optional[Dict[str, str]] = None) -> List[str]:
        from google.cloud import compute_v1
        
        request = compute_v1.ListInstancesRequest()
        request.project = self.project
        request.zone = self.zone
        
        label_filter = 'labels.axonet-job="true"'
        if labels:
            for k, v in labels.items():
                label_filter += f' AND labels.{k.replace("_", "-")}="{v}"'
        request.filter = label_filter
        
        instances = self.client.list(request=request)
        return [instance.name for instance in instances]

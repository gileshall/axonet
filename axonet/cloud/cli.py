#!/usr/bin/env python3
"""CLI for submitting cloud jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .base import JobConfig, JobStatus
from .registry import get_provider, list_providers


def _get_gcp_env() -> dict:
    """Get GCP environment variables to pass to jobs."""
    env = {}
    for var in ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_REGION", "GOOGLE_CLOUD_ZONE", "AXONET_GCS_BUCKET"]:
        if os.environ.get(var):
            env[var] = os.environ[var]
    return env


def cmd_generate_dataset(args):
    """Submit dataset generation job(s)."""
    provider = get_provider(args.provider)
    
    manifest_entries = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_entries.append(json.loads(line))
    
    # Apply limit for smoke tests
    if args.limit and args.limit < len(manifest_entries):
        manifest_entries = manifest_entries[:args.limit]
        print(f"Limited to {args.limit} neurons (smoke test)")
    
    total = len(manifest_entries)
    num_tasks = min(args.parallelism, total)
    
    print(f"Submitting dataset generation: {total} neurons, {num_tasks} tasks")
    
    if args.upload_manifest:
        remote_manifest = f"{args.output.rstrip('/')}/input/manifest.jsonl"
        # If limited, write a temp manifest with only the limited entries
        if args.limit:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for entry in manifest_entries:
                    f.write(json.dumps(entry) + "\n")
                tmp_manifest = f.name
            provider.storage.upload(Path(tmp_manifest), remote_manifest)
            os.unlink(tmp_manifest)
        else:
            provider.storage.upload(Path(args.manifest), remote_manifest)
        manifest_arg = remote_manifest
    else:
        manifest_arg = args.manifest
    
    # Pass GCP environment variables to container
    env = _get_gcp_env()
    
    configs = []
    for i in range(num_tasks):
        # Only pass arguments - the container ENTRYPOINT handles the python command
        cmd_args = [
            "--manifest", manifest_arg,
            "--swc-prefix", args.swc_prefix,
            "--output", args.output,
            "--width", str(args.width),
            "--height", str(args.height),
            "--views", str(args.views),
            "--task-index", str(i),
            "--total-tasks", str(num_tasks),
            "--provider", args.provider,
        ]
        if args.no_cache:
            cmd_args.append("--no-cache")
        
        config = JobConfig(
            name=f"dataset-gen-{i}",
            command=cmd_args,
            image=args.image,
            cpu=args.cpu,
            memory_gb=args.memory,
            gpu_type=args.gpu_type if args.gpu_count > 0 else None,
            gpu_count=args.gpu_count,
            disk_gb=args.disk,
            max_runtime_hours=args.timeout,
            env=env,
            labels={"job-type": "dataset-generation"},
            machine_type=args.machine_type,
        )
        configs.append(config)
    
    batch_id = provider.batch.submit_array(configs, parallelism=args.parallelism)
    print(f"Submitted batch: {batch_id}")
    
    if hasattr(provider.batch, "get_logs_url"):
        print(f"Logs: {provider.batch.get_logs_url(batch_id)}")
    
    if args.wait:
        print("Waiting for completion...")
        results = provider.batch.wait(batch_id)
        
        succeeded = sum(1 for r in results.values() if r.status == JobStatus.SUCCEEDED)
        failed = sum(1 for r in results.values() if r.status == JobStatus.FAILED)
        
        print(f"Completed: {succeeded} succeeded, {failed} failed")
        
        if failed > 0:
            sys.exit(1)


def cmd_train(args):
    """Submit training job."""
    provider = get_provider(args.provider)
    
    print(f"Submitting Stage {args.stage} training job")
    
    # Pass GCP environment variables to job
    env = _get_gcp_env()
    
    command = [
        "python", "-m", "axonet.cloud.entrypoints.train",
        "--stage", str(args.stage),
        "--data-dir", args.data_dir,
        "--manifest", args.manifest,
        "--output", args.output,
        "--provider", args.provider,
    ]
    
    if args.config:
        provider.storage.upload(Path(args.config), f"{args.output.rstrip('/')}/config.yaml")
        command.extend(["--config", f"{args.output.rstrip('/')}/config.yaml"])
    
    if args.stage == 2:
        if not args.stage1_checkpoint:
            print("ERROR: --stage1-checkpoint required for Stage 2")
            sys.exit(1)
        command.extend(["--stage1-checkpoint", args.stage1_checkpoint])
        if args.metadata:
            command.extend(["--metadata", args.metadata])
    
    if args.batch_size:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.lr:
        command.extend(["--lr", str(args.lr)])
    if args.max_epochs:
        command.extend(["--max-epochs", str(args.max_epochs)])
    if args.precision:
        command.extend(["--precision", args.precision])
    
    config = JobConfig(
        name=f"train-stage{args.stage}",
        command=command,
        image=args.image,
        cpu=args.cpu,
        memory_gb=args.memory,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        disk_gb=args.disk,
        max_runtime_hours=args.timeout,
        env=env,
        labels={"job-type": "training", "stage": str(args.stage)},
        machine_type=args.machine_type,
    )
    
    job_id = provider.compute.submit(config)
    print(f"Submitted job: {job_id}")
    
    if args.wait:
        print("Waiting for completion...")
        result = provider.compute.wait(job_id)
        print(f"Status: {result.status.value}")
        
        if result.status == JobStatus.FAILED:
            print("Job failed. Logs:")
            print(result.logs or "(no logs available)")
            sys.exit(1)


def cmd_status(args):
    """Check job status."""
    provider = get_provider(args.provider)
    
    if args.batch:
        statuses = provider.batch.status(args.job_id)
        for task_id, status in statuses.items():
            print(f"{task_id}: {status.value}")
    else:
        status = provider.compute.status(args.job_id)
        print(f"{args.job_id}: {status.value}")


def cmd_logs(args):
    """Get job logs."""
    provider = get_provider(args.provider)
    
    logs = provider.compute.logs(args.job_id)
    print(logs)


def cmd_cancel(args):
    """Cancel a job."""
    provider = get_provider(args.provider)
    
    if args.batch:
        provider.batch.cancel(args.job_id)
    else:
        provider.compute.cancel(args.job_id)
    
    print(f"Cancelled: {args.job_id}")


def cmd_list(args):
    """List jobs."""
    provider = get_provider(args.provider)
    
    labels = {}
    if args.job_type:
        labels["job-type"] = args.job_type
    
    jobs = provider.compute.list_jobs(labels=labels if labels else None)
    
    for job_id in jobs:
        status = provider.compute.status(job_id)
        print(f"{job_id}: {status.value}")


def main():
    parser = argparse.ArgumentParser(description="axonet cloud CLI")
    parser.add_argument("--provider", default="google", choices=list_providers())
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    gen_parser = subparsers.add_parser("generate-dataset", help="Generate training dataset")
    gen_parser.add_argument("--manifest", required=True, help="Input manifest JSONL")
    gen_parser.add_argument("--swc-prefix", required=True, help="SWC files location")
    gen_parser.add_argument("--output", required=True, help="Output location (gs://...)")
    gen_parser.add_argument("--upload-manifest", action="store_true", help="Upload manifest to cloud")
    gen_parser.add_argument("--width", type=int, default=512)
    gen_parser.add_argument("--height", type=int, default=512)
    gen_parser.add_argument("--views", type=int, default=24)
    gen_parser.add_argument("--no-cache", action="store_true")
    gen_parser.add_argument("--parallelism", type=int, default=10)
    gen_parser.add_argument("--limit", type=int, help="Limit neurons (smoke test)")
    gen_parser.add_argument("--image", default="gcr.io/PROJECT/axonet:latest")
    gen_parser.add_argument("--cpu", type=int, default=4)
    gen_parser.add_argument("--memory", type=int, default=16)
    gen_parser.add_argument("--gpu-type", default=None, help="GPU type: t4, l4, v100, a100")
    gen_parser.add_argument("--gpu-count", type=int, default=0, help="GPUs per task")
    gen_parser.add_argument("--machine-type", default=None, help="Explicit machine type (e.g., n1-standard-8)")
    gen_parser.add_argument("--disk", type=int, default=100)
    gen_parser.add_argument("--timeout", type=float, default=6.0, help="Hours")
    gen_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    gen_parser.set_defaults(func=cmd_generate_dataset)
    
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", type=Path, help="Config YAML")
    train_parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    train_parser.add_argument("--data-dir", required=True, help="Dataset location")
    train_parser.add_argument("--manifest", default="manifest.jsonl")
    train_parser.add_argument("--metadata", default="metadata.jsonl")
    train_parser.add_argument("--output", required=True, help="Output location")
    train_parser.add_argument("--stage1-checkpoint", help="Stage 1 checkpoint for Stage 2")
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--lr", type=float)
    train_parser.add_argument("--max-epochs", type=int)
    train_parser.add_argument("--precision", default="32")
    train_parser.add_argument("--image", default="gcr.io/PROJECT/axonet:latest")
    train_parser.add_argument("--cpu", type=int, default=8)
    train_parser.add_argument("--memory", type=int, default=32)
    train_parser.add_argument("--gpu-type", default="t4")
    train_parser.add_argument("--gpu-count", type=int, default=1)
    train_parser.add_argument("--machine-type", default=None, help="Explicit machine type")
    train_parser.add_argument("--disk", type=int, default=200)
    train_parser.add_argument("--timeout", type=float, default=24.0, help="Hours")
    train_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    train_parser.set_defaults(func=cmd_train)
    
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("job_id", help="Job or batch ID")
    status_parser.add_argument("--batch", action="store_true", help="ID is a batch ID")
    status_parser.set_defaults(func=cmd_status)
    
    logs_parser = subparsers.add_parser("logs", help="Get job logs")
    logs_parser.add_argument("job_id", help="Job ID")
    logs_parser.set_defaults(func=cmd_logs)
    
    cancel_parser = subparsers.add_parser("cancel", help="Cancel job")
    cancel_parser.add_argument("job_id", help="Job or batch ID")
    cancel_parser.add_argument("--batch", action="store_true", help="ID is a batch ID")
    cancel_parser.set_defaults(func=cmd_cancel)
    
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--job-type", choices=["dataset-generation", "training"])
    list_parser.set_defaults(func=cmd_list)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

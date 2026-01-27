#!/usr/bin/env python3
"""CLI for submitting cloud jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
    
    if args.manifest:
        # Load from provided manifest
        with open(args.manifest) as f:
            for line in f:
                line = line.strip()
                if line:
                    manifest_entries.append(json.loads(line))
    else:
        # Auto-discover SWC files from prefix
        print(f"Discovering SWC files from {args.swc_prefix}...")
        swc_files = provider.storage.list_files(args.swc_prefix)
        swc_files = [f for f in swc_files if f.lower().endswith('.swc')]
        print(f"Found {len(swc_files)} SWC files")
        
        for swc_path in swc_files:
            filename = swc_path.rsplit('/', 1)[-1]
            neuron_id = filename.rsplit('.', 1)[0]
            manifest_entries.append({
                "neuron_id": neuron_id,
                "swc": filename,
            })
    
    # Apply limit for smoke tests
    if args.limit and args.limit < len(manifest_entries):
        manifest_entries = manifest_entries[:args.limit]
        print(f"Limited to {args.limit} neurons (smoke test)")
    
    total = len(manifest_entries)
    if total == 0:
        print("ERROR: No SWC files found")
        sys.exit(1)
    
    num_tasks = min(args.parallelism, total)
    
    print(f"Submitting dataset generation: {total} neurons, {num_tasks} tasks")
    
    # Always upload manifest (auto-generated or provided)
    remote_manifest = f"{args.output.rstrip('/')}/input/manifest.jsonl"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")
        tmp_manifest = f.name
    provider.storage.upload(Path(tmp_manifest), remote_manifest)
    os.unlink(tmp_manifest)
    manifest_arg = remote_manifest
    
    # Pass GCP environment variables to container
    env = _get_gcp_env()
    
    # Build command args once - Google Batch sets BATCH_TASK_INDEX per task
    cmd_args = [
        "--manifest", manifest_arg,
        "--swc-prefix", args.swc_prefix,
        "--output", args.output,
        "--width", str(args.width),
        "--height", str(args.height),
        "--views", str(args.views),
        "--segments", str(args.segments),
        "--supersample-factor", str(args.supersample_factor),
        "--margin", str(args.margin),
        "--projection", args.projection,
        "--fovy", str(args.fovy),
        "--min-qc", str(args.min_qc),
        "--qc-retries", str(args.qc_retries),
        "--radius-scale", str(args.radius_scale),
        "--radius-adaptive-alpha", str(args.radius_adaptive_alpha),
        "--radius-ref-percentile", str(args.radius_ref_percentile),
        "--seed", str(args.seed),
        "--bg", *[str(c) for c in args.bg],
        "--total-tasks", str(num_tasks),
        "--provider", args.provider,
        "--sampling", args.sampling,
    ]
    if args.depth_shading:
        cmd_args.append("--depth-shading")
    if args.adaptive_framing:
        cmd_args.append("--adaptive-framing")
    if args.auto_margin and not args.no_auto_margin:
        cmd_args.append("--auto-margin")
    if args.no_cache:
        cmd_args.append("--no-cache")
    if args.save_cache:
        cmd_args.append("--save-cache")

    configs = []
    for i in range(num_tasks):
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

        # Merge per-task manifests into a single manifest_train.jsonl
        print("Merging per-task manifests...")
        merged_entries = []
        for i in range(num_tasks):
            part_path = f"{args.output.rstrip('/')}/manifests/manifest_{i}.jsonl"
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
                    tmp_path = tmp.name
                provider.storage.download(part_path, Path(tmp_path))
                with open(tmp_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            merged_entries.append(line)
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Warning: could not download manifest part {i}: {e}")

        if merged_entries:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for entry in merged_entries:
                    f.write(entry + "\n")
                tmp_manifest = f.name
            remote_manifest_train = f"{args.output.rstrip('/')}/manifest_train.jsonl"
            provider.storage.upload(Path(tmp_manifest), remote_manifest_train)
            os.unlink(tmp_manifest)
            print(f"Merged {len(merged_entries)} entries into {remote_manifest_train}")
        else:
            print("Warning: no manifest entries found to merge")


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
    gen_parser.add_argument("--manifest", help="Input manifest JSONL (auto-discovers from swc-prefix if omitted)")
    gen_parser.add_argument("--swc-prefix", required=True, help="SWC files location")
    gen_parser.add_argument("--output", required=True, help="Output location (gs://...)")
    gen_parser.add_argument("--width", type=int, default=512)
    gen_parser.add_argument("--height", type=int, default=512)
    gen_parser.add_argument("--views", type=int, default=24)
    gen_parser.add_argument("--segments", type=int, default=32, help="Mesh segments per cylinder")
    gen_parser.add_argument("--supersample-factor", type=int, default=2, help="Supersampling factor")
    gen_parser.add_argument("--margin", type=float, default=0.40, help="Camera margin around neuron")
    gen_parser.add_argument("--projection", choices=["ortho", "persp"], default="ortho")
    gen_parser.add_argument("--fovy", type=float, default=55.0, help="Field of view in degrees")
    gen_parser.add_argument("--depth-shading", action="store_true", help="Enable depth shading")
    gen_parser.add_argument("--bg", type=float, nargs=4, default=(0, 0, 0, 1), metavar=("R", "G", "B", "A"),
                           help="Background color RGBA")
    gen_parser.add_argument("--min-qc", type=float, default=0.7, help="Minimum QC fraction")
    gen_parser.add_argument("--qc-retries", type=int, default=5, help="QC retry attempts per view")
    gen_parser.add_argument("--radius-scale", type=float, default=1.0, help="Global radius scale factor")
    gen_parser.add_argument("--radius-adaptive-alpha", type=float, default=0.0,
                           help="Adaptive radius scaling strength (0 disables)")
    gen_parser.add_argument("--radius-ref-percentile", type=float, default=50.0,
                           help="Reference percentile for adaptive radius scaling")
    gen_parser.add_argument("--auto-margin", action="store_true", help="Auto-expand margin if needed")
    gen_parser.add_argument("--no-auto-margin", action="store_true", help="Disable auto-margin")
    gen_parser.add_argument("--sampling", choices=["pca", "fibonacci", "random"], default="pca",
                           help="Camera sampling strategy (default: pca)")
    gen_parser.add_argument("--adaptive-framing", action="store_true",
                           help="Per-view adaptive ortho_scale based on projected extent")
    gen_parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    gen_parser.add_argument("--no-cache", action="store_true")
    gen_parser.add_argument("--save-cache", action="store_true", help="Upload mesh cache for later reuse")
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

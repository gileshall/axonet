"""CLI tool for linear probing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .linear_probing import run_linear_probing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Linear probing for SWC-VAE embeddings")
    
    parser.add_argument("--vae-checkpoint", type=Path, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for results")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--swc-dir", type=Path, help="Directory containing SWC files")
    input_group.add_argument("--manifest", type=Path, help="Path to manifest.jsonl")
    
    parser.add_argument("--data-dir", type=Path, help="Data root directory (required if using manifest)")
    parser.add_argument("--swc-root", type=Path, help="Root directory containing SWC files (required if using manifest)")
    parser.add_argument("--metadata-csv", type=Path, help="Path to PatchSeq_metadata.csv")
    
    parser.add_argument("--tasks", type=str, default="ttype,met,layer,depth",
                       help="Comma-separated list of tasks (default: ttype,met,layer,depth)")
    parser.add_argument("--probe-type", choices=["linear", "mlp"], default="linear",
                       help="Probe architecture type")
    parser.add_argument("--use-norm", action="store_true",
                       help="Use LayerNorm before probe heads (for linear probes)")
    parser.add_argument("--hidden-dim", type=int, default=None,
                       help="Hidden dimension for MLP probe")
    
    parser.add_argument("--n-splits", type=int, default=5,
                       help="Number of CV folds (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Maximum number of epochs (default: 100)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                       help="Weight decay (default: 1e-2)")
    parser.add_argument("--task-weights", type=str, default=None,
                       help="Comma-separated task weights (e.g., 'ttype:1.0,met:1.0,layer:1.0,depth:1.0')")
    parser.add_argument("--no-class-weights", action="store_true",
                       help="Disable class weights for imbalanced classes")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                       help="Early stopping patience (default: 10)")
    parser.add_argument("--early-stopping-metric", choices=["macro_f1", "accuracy", "r2"], default="macro_f1",
                       help="Metric to monitor for early stopping (default: macro_f1)")
    
    parser.add_argument("--use-mu", action="store_true", default=True,
                       help="Use mu (deterministic) instead of z (stochastic) for embeddings (default: True)")
    parser.add_argument("--use-z", dest="use_mu", action="store_false",
                       help="Use z (stochastic) instead of mu (deterministic)")
    parser.add_argument("--embedding-reduce", choices=["mean", "max", "flatten"], default="mean",
                       help="How to reduce spatial dimensions (default: mean)")
    parser.add_argument("--width", type=int, default=1024,
                       help="Render width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024,
                       help="Render height (default: 1024)")
    
    parser.add_argument("--pose-sampling", choices=["random", "all", "first"], default="random",
                       help="How to sample poses (default: random)")
    parser.add_argument("--n-poses-per-neuron", type=int, default=1,
                       help="Number of poses per neuron if pose_sampling=random (default: 1)")
    
    parser.add_argument("--no-pose-stats", action="store_true",
                       help="Skip pose statistics computation")
    parser.add_argument("--no-depth-diag", action="store_true",
                       help="Skip depth diagnostics")
    
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto/mps/cuda/cpu, default: auto)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    
    parser.add_argument("--pattern", type=str, default="*.swc",
                       help="Glob pattern for SWC files (default: *.swc)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    if args.manifest and not args.data_dir:
        parser.error("--data-dir is required when using --manifest")
    if args.manifest and not args.swc_root:
        parser.error("--swc-root is required when using --manifest")
    
    tasks = [t.strip() for t in args.tasks.split(",")]
    
    task_weights = None
    if args.task_weights:
        task_weights = {}
        for pair in args.task_weights.split(","):
            task, weight = pair.split(":")
            task_weights[task.strip()] = float(weight.strip())
    
    swc_paths = None
    if args.swc_dir:
        swc_paths = sorted(args.swc_dir.glob(args.pattern))
        if not swc_paths:
            logger.error(f"No SWC files found in {args.swc_dir} matching pattern {args.pattern}")
            return
        logger.info(f"Found {len(swc_paths)} SWC files")
    
    results = run_linear_probing(
        vae_checkpoint=args.vae_checkpoint,
        swc_paths=swc_paths,
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        swc_root=args.swc_root,
        metadata_csv=args.metadata_csv,
        output_dir=args.output,
        tasks=tasks,
        probe_type=args.probe_type,
        use_norm=args.use_norm,
        hidden_dim=args.hidden_dim,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        task_weights=task_weights,
        use_class_weights=not args.no_class_weights,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        use_mu=args.use_mu,
        embedding_reduce=args.embedding_reduce,
        width=args.width,
        height=args.height,
        pose_sampling=args.pose_sampling,
        n_poses_per_neuron=args.n_poses_per_neuron,
        compute_pose_stats=not args.no_pose_stats,
        compute_depth_diag=not args.no_depth_diag,
        device=args.device,
        seed=args.seed,
    )
    
    logger.info("\nFinal Results:")
    for task in tasks:
        if task in results["aggregated_metrics"]:
            metrics = results["aggregated_metrics"][task]
            if task == "depth":
                logger.info(f"  {task}: R² = {metrics.get('r2', {}).get('mean', 0):.4f} ± {metrics.get('r2', {}).get('std', 0):.4f}")
            else:
                logger.info(f"  {task}: Accuracy = {metrics.get('accuracy', {}).get('mean', 0):.4f} ± {metrics.get('accuracy', {}).get('std', 0):.4f}")
                logger.info(f"         Macro-F1 = {metrics.get('macro_f1', {}).get('mean', 0):.4f} ± {metrics.get('macro_f1', {}).get('std', 0):.4f}")


if __name__ == "__main__":
    main()


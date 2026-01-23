#!/bin/bash
# Example end-to-end workflow for training on Google Cloud
# This script demonstrates the full pipeline from data to trained model
#
# Setup options:
#   1. With service account (requires IAM permissions):
#      ./setup.sh PROJECT_ID REGION
#
#   2. Without service account (uses default compute SA):
#      ./setup.sh --no-service-account PROJECT_ID REGION
#
#   3. Minimal (just bucket, for users without admin access):
#      ./setup.sh --bucket-only PROJECT_ID REGION
#      # Then ask admin to grant you Storage Admin on the bucket

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
BUCKET="${AXONET_GCS_BUCKET:-${PROJECT_ID}-axonet}"
REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/axonet"

echo "================================================"
echo "Axonet Google Cloud Training Workflow"
echo "================================================"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: $BUCKET"
echo ""
echo "NOTE: If AXONET_SERVICE_ACCOUNT is not set, jobs will use"
echo "      the project's default compute service account."
echo ""

# Step 1: Upload source data to GCS
echo "Step 1: Upload source data"
echo "  gsutil -m cp -r local_data/swc gs://${BUCKET}/neuromorpho/swc/"
echo "  gsutil cp local_data/metadata.jsonl gs://${BUCKET}/neuromorpho/metadata.jsonl"
echo ""

# Step 2: Generate dataset using Google Batch
echo "Step 2: Generate dataset (Batch)"
cat <<EOF
  axonet-cloud --provider google generate-dataset \\
    --manifest local_data/metadata.jsonl \\
    --upload-manifest \\
    --swc-prefix gs://${BUCKET}/neuromorpho/swc \\
    --output gs://${BUCKET}/datasets/v1 \\
    --parallelism 20 \\
    --image ${REPO}/axonet-dataset:latest \\
    --cpu 4 \\
    --memory 16 \\
    --wait
EOF
echo ""

# Step 3: Merge manifests from batch tasks
echo "Step 3: Merge manifests"
cat <<EOF
  gsutil cat gs://${BUCKET}/datasets/v1/manifests/*.jsonl > manifest.jsonl
  gsutil cp manifest.jsonl gs://${BUCKET}/datasets/v1/manifest.jsonl
EOF
echo ""

# Step 4: Train Stage 1 VAE
echo "Step 4: Train Stage 1 (Compute Engine)"
cat <<EOF
  axonet-cloud --provider google train \\
    --stage 1 \\
    --data-dir gs://${BUCKET}/datasets/v1 \\
    --manifest manifest.jsonl \\
    --output gs://${BUCKET}/models/stage1 \\
    --image ${REPO}/axonet-train:latest \\
    --gpu-type t4 \\
    --gpu-count 1 \\
    --max-epochs 100 \\
    --wait
EOF
echo ""

# Step 5: Train Stage 2 CLIP
echo "Step 5: Train Stage 2 CLIP (Compute Engine)"
cat <<EOF
  axonet-cloud --provider google train \\
    --stage 2 \\
    --data-dir gs://${BUCKET}/datasets/v1 \\
    --manifest manifest.jsonl \\
    --metadata metadata.jsonl \\
    --stage1-checkpoint gs://${BUCKET}/models/stage1/checkpoints/best.ckpt \\
    --output gs://${BUCKET}/models/stage2 \\
    --image ${REPO}/axonet-train:latest \\
    --gpu-type t4 \\
    --gpu-count 1 \\
    --max-epochs 50 \\
    --wait
EOF
echo ""

# Step 6: Download trained model
echo "Step 6: Download trained model"
cat <<EOF
  gsutil -m cp -r gs://${BUCKET}/models/stage2/checkpoints ./checkpoints/
EOF
echo ""

echo "================================================"
echo "Workflow complete!"
echo "================================================"

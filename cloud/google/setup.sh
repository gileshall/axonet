#!/bin/bash
# Setup script for Google Cloud infrastructure
# Usage: ./setup.sh [OPTIONS] [PROJECT_ID] [REGION] [BUCKET_NAME]
#
# Options:
#   --no-service-account    Skip service account creation (use user credentials)
#   --skip-iam              Skip IAM policy bindings (if you lack permissions)
#   --bucket-only           Only create the bucket, skip everything else

set -e

# Parse flags
SETUP_SERVICE_ACCOUNT=true
SETUP_IAM=true
BUCKET_ONLY=false

while [[ "$1" == --* ]]; do
    case "$1" in
        --no-service-account)
            SETUP_SERVICE_ACCOUNT=false
            shift
            ;;
        --skip-iam)
            SETUP_IAM=false
            shift
            ;;
        --bucket-only)
            BUCKET_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-us-central1}"
BUCKET_NAME="${3:-${PROJECT_ID}-axonet}"

echo "Setting up axonet infrastructure..."
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Bucket: $BUCKET_NAME"
echo "  Service Account: $SETUP_SERVICE_ACCOUNT"
echo "  IAM Bindings: $SETUP_IAM"
echo ""

gcloud config set project "$PROJECT_ID"

if [ "$BUCKET_ONLY" = false ]; then
    echo "Enabling required APIs..."
    gcloud services enable \
        batch.googleapis.com \
        compute.googleapis.com \
        storage.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com \
        cloudbuild.googleapis.com
fi

echo "Creating GCS bucket..."
gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$BUCKET_NAME" 2>/dev/null || echo "Bucket already exists"

TMP_FILE=$(mktemp)
cat > $TMP_FILE <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30, "matchesPrefix": ["jobs/", "tmp/"]}
      }
    ]
  }
}
EOF
gsutil lifecycle set $TMP_FILE "gs://$BUCKET_NAME" || echo "Could not set lifecycle (may need bucket admin permission)"
rm $TMP_FILE

if [ "$BUCKET_ONLY" = true ]; then
    echo ""
    echo "Bucket-only setup complete!"
    echo "  export AXONET_GCS_BUCKET=$BUCKET_NAME"
    exit 0
fi

SA_EMAIL=""

if [ "$SETUP_SERVICE_ACCOUNT" = true ]; then
    echo "Creating service account..."
    SA_NAME="axonet-worker"
    SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

    gcloud iam service-accounts create "$SA_NAME" \
        --display-name "Axonet Worker" \
        --description "Service account for axonet batch and compute jobs" \
        2>/dev/null || echo "Service account already exists"

    if [ "$SETUP_IAM" = true ]; then
        echo "Granting permissions..."
        echo "(If this fails, ask your project admin to grant these roles, or use --skip-iam)"
        
        ROLES=(
            "roles/storage.objectAdmin"
            "roles/batch.jobsEditor"
            "roles/compute.instanceAdmin.v1"
            "roles/logging.logWriter"
        )
        
        for ROLE in "${ROLES[@]}"; do
            echo "  Granting $ROLE..."
            if ! gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member "serviceAccount:$SA_EMAIL" \
                --role "$ROLE" \
                --quiet 2>/dev/null; then
                echo "    WARNING: Could not grant $ROLE (insufficient permissions)"
                echo "    Ask your admin to run:"
                echo "      gcloud projects add-iam-policy-binding $PROJECT_ID \\"
                echo "        --member serviceAccount:$SA_EMAIL --role $ROLE"
            fi
        done
    else
        echo ""
        echo "Skipping IAM bindings. Ask your project admin to grant these roles:"
        echo "  gcloud projects add-iam-policy-binding $PROJECT_ID \\"
        echo "    --member serviceAccount:$SA_EMAIL \\"
        echo "    --role roles/storage.objectAdmin"
        echo ""
        echo "  (Also: roles/batch.jobsEditor, roles/compute.instanceAdmin.v1, roles/logging.logWriter)"
    fi
fi

echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create axonet \
    --repository-format docker \
    --location "$REGION" \
    --description "Axonet container images" \
    2>/dev/null || echo "Repository already exists"

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Environment variables to set:"
echo "  export GOOGLE_CLOUD_PROJECT=$PROJECT_ID"
echo "  export GOOGLE_CLOUD_REGION=$REGION"
echo "  export GOOGLE_CLOUD_ZONE=${REGION}-a"
echo "  export AXONET_GCS_BUCKET=$BUCKET_NAME"

if [ -n "$SA_EMAIL" ]; then
    echo "  export AXONET_SERVICE_ACCOUNT=$SA_EMAIL"
fi

echo ""

if [ "$SETUP_SERVICE_ACCOUNT" = false ]; then
    echo "NOTE: Running without service account."
    echo "Jobs will use your user credentials via Application Default Credentials."
    echo "Make sure you have authenticated with:"
    echo "  gcloud auth application-default login"
    echo ""
fi

echo "To build and push Docker image:"
echo "  ./cloud/google/build.sh"

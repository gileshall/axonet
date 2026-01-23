#!/bin/bash
# Build and push Docker images to Artifact Registry
# Usage: ./build.sh [TAG] [--local]
#
# Options:
#   TAG         Image tag (default: latest)
#   --local     Build for local architecture only (skip cross-compile)
#   --no-push   Build but don't push to registry

set -e

# Parse args
TAG="latest"
LOCAL_ONLY=false
NO_PUSH=false

for arg in "$@"; do
    case "$arg" in
        --local)
            LOCAL_ONLY=true
            ;;
        --no-push)
            NO_PUSH=true
            ;;
        *)
            TAG="$arg"
            ;;
    esac
done

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/axonet"

# Detect platform - GCP runs on x86_64
PLATFORM=""
BUILD_CMD="docker build"

if [ "$LOCAL_ONLY" = false ]; then
    # Check if we're on ARM (Apple Silicon Mac)
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        echo "Detected ARM architecture - building for linux/amd64 (GCP target)"
        PLATFORM="--platform linux/amd64"
        
        # Use buildx for cross-platform builds
        if docker buildx version &>/dev/null; then
            # Ensure buildx builder exists
            if ! docker buildx inspect axonet-builder &>/dev/null; then
                echo "Creating buildx builder for cross-platform builds..."
                docker buildx create --name axonet-builder --use
            else
                docker buildx use axonet-builder
            fi
            BUILD_CMD="docker buildx build --load"
        else
            echo "WARNING: docker buildx not available, using standard build with QEMU emulation"
            echo "         This may be slow. Consider installing Docker Desktop or buildx."
        fi
    fi
fi

echo "Building axonet images..."
echo "  Repository: $REPO"
echo "  Tag: $TAG"
echo "  Platform: ${PLATFORM:-native}"
echo ""

if [ "$NO_PUSH" = false ]; then
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
fi

echo "Building dataset image..."
$BUILD_CMD $PLATFORM --target dataset -t "${REPO}/axonet-dataset:${TAG}" .

echo "Building training image..."
$BUILD_CMD $PLATFORM --target train -t "${REPO}/axonet-train:${TAG}" .

echo "Building production image..."
$BUILD_CMD $PLATFORM --target prod -t "${REPO}/axonet:${TAG}" .

if [ "$NO_PUSH" = true ]; then
    echo ""
    echo "Images built (not pushed):"
    echo "  ${REPO}/axonet-dataset:${TAG}"
    echo "  ${REPO}/axonet-train:${TAG}"
    echo "  ${REPO}/axonet:${TAG}"
    exit 0
fi

echo "Pushing images..."
docker push "${REPO}/axonet-dataset:${TAG}"
docker push "${REPO}/axonet-train:${TAG}"
docker push "${REPO}/axonet:${TAG}"

echo ""
echo "Images pushed:"
echo "  ${REPO}/axonet-dataset:${TAG}"
echo "  ${REPO}/axonet-train:${TAG}"
echo "  ${REPO}/axonet:${TAG}"
echo ""
echo "Use with axonet-cloud:"
echo "  axonet-cloud generate-dataset --image ${REPO}/axonet-dataset:${TAG} ..."
echo "  axonet-cloud train --image ${REPO}/axonet-train:${TAG} ..."

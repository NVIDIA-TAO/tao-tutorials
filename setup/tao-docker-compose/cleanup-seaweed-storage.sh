#!/usr/bin/env bash
# Clean up SeaweedFS S3 storage (tao-storage bucket).
# Deletes all objects (datasets, shared-storage/models, job outputs). Does NOT clear workspaces:
# workspace/experiment/job metadata live in MongoDB. To clear those run: ./run.sh clear-mongo

set -e

SEAWEED_ENDPOINT="${SEAWEED_ENDPOINT:-http://localhost:8333}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-seaweedfs}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-seaweedfs123}"
BUCKET="${TAO_STORAGE_BUCKET:-tao-storage}"

echo "Seaweed cleanup: endpoint=$SEAWEED_ENDPOINT bucket=$BUCKET"
if ! command -v aws &>/dev/null; then
    echo "Error: aws CLI not found. Install it (e.g. pip install awscli) and retry."
    exit 1
fi

echo "Listing current objects..."
if ! AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    aws s3 ls "s3://${BUCKET}/" --endpoint-url "$SEAWEED_ENDPOINT" --recursive 2>/dev/null | head -50; then
    echo "Bucket may be empty or unreachable. Attempting delete anyway."
fi

echo "Deleting all objects in s3://${BUCKET}/ ..."
AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    aws s3 rm "s3://${BUCKET}/" --endpoint-url "$SEAWEED_ENDPOINT" --recursive

echo "Seaweed storage cleanup completed. Re-upload data and run load_airgapped_model as needed."
echo "To clear older workspaces/experiments (stored in MongoDB), run: ./run.sh clear-mongo"

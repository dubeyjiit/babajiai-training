set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
STAGING_BUCKET="${STAGING_BUCKET:?Set STAGING_BUCKET (gs://...)}"
SA_EMAIL="vertex-mlops-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# These come from the last successful training pipeline run
ENDPOINT_ID="${PRODUCTION_ENDPOINT_ID:-${ENDPOINT_ID:-}}"
MODEL_RESOURCE_NAME="${MODEL_RESOURCE_NAME:-}"
PREPROCESSOR_GCS_PATH="${PREPROCESSOR_GCS_PATH:-}"

COMPILED_YAML_GCS="${STAGING_BUCKET}/compiled_pipelines/inference_pipeline.yaml"
FUNCTION_NAME="bq-inference-trigger"
SCHEDULER_JOB_NAME="inference-pipeline-scheduler"
SCHEDULE="${SCHEDULE:-*/15 * * * *}"   # every 15 minutes

echo "============================================================"
echo " Deploying BQ → Inference Pipeline trigger"
echo "   Function: ${FUNCTION_NAME}"
echo "   Schedule: ${SCHEDULE}"
echo "============================================================"

# ── Deploy Cloud Function ──────────────────────────────────────────────────────
echo ""
echo "=== Deploying Cloud Function ==="
gcloud functions deploy "${FUNCTION_NAME}" \
  --gen2 \
  --runtime=python311 \
  --region="${REGION}" \
  --source="$(dirname "$0")/bq_trigger_function" \
  --entry-point=trigger_inference_pipeline \
  --trigger-http \
  --allow-unauthenticated \
  --service-account="${SA_EMAIL}" \
  --timeout=540s \
  --memory=512Mi \
  --set-env-vars="\
PROJECT_ID=${PROJECT_ID},\
REGION=${REGION},\
STAGING_BUCKET=${STAGING_BUCKET},\
COMPILED_INFERENCE_YAML_GCS=${COMPILED_YAML_GCS},\
PRODUCTION_ENDPOINT_ID=${ENDPOINT_ID},\
MODEL_RESOURCE_NAME=${MODEL_RESOURCE_NAME},\
PREPROCESSOR_GCS_PATH=${PREPROCESSOR_GCS_PATH},\
BATCH_THRESHOLD=100" \
  --project="${PROJECT_ID}"

# Get function URL
FUNCTION_URL=$(gcloud functions describe "${FUNCTION_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(url)")
echo "Function deployed: ${FUNCTION_URL}"

# ── Create Cloud Scheduler job ─────────────────────────────────────────────────
echo ""
echo "=== Creating Cloud Scheduler job ==="
# Delete if exists
gcloud scheduler jobs delete "${SCHEDULER_JOB_NAME}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet 2>/dev/null || true

gcloud scheduler jobs create http "${SCHEDULER_JOB_NAME}" \
  --location="${REGION}" \
  --schedule="${SCHEDULE}" \
  --uri="${FUNCTION_URL}" \
  --http-method=POST \
  --oidc-service-account-email="${SA_EMAIL}" \
  --oidc-token-audience="${FUNCTION_URL}" \
  --time-zone="America/New_York" \
  --project="${PROJECT_ID}"

echo "Scheduler job created: ${SCHEDULER_JOB_NAME}"
echo ""
echo "=== Trigger deployment complete ==="
echo "The inference pipeline will run every: ${SCHEDULE}"
echo ""
echo "To update env vars after a new training run:"
echo "  gcloud functions deploy ${FUNCTION_NAME} --gen2 \\"
echo "    --update-env-vars=PRODUCTION_ENDPOINT_ID=<new_endpoint>,MODEL_RESOURCE_NAME=<new_model>,..."
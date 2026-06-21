set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SA_NAME="vertex-mlops-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET="${PROJECT_ID}-mlops-artifacts"
BQ_DATASET="used_car_mlops"

echo "============================================================"
echo " Setting up MLOps GCP resources"
echo "   Project:  ${PROJECT_ID}"
echo "   Region:   ${REGION}"
echo "   Bucket:   gs://${BUCKET}"
echo "   SA:       ${SA_EMAIL}"
echo "============================================================"

echo ""
echo "=== 1. Enabling APIs ==="
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  bigquerystorage.googleapis.com \
  cloudbuild.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudscheduler.googleapis.com \
  eventarc.googleapis.com \
  iam.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT_ID}"
echo "APIs enabled."

echo ""
echo ""
echo "=== 2. Creating GCS bucket ==="
if ! gcloud storage ls "gs://${BUCKET}" &>/dev/null; then
  gcloud storage buckets create "gs://${BUCKET}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}"
  echo "Bucket created: gs://${BUCKET}"
else
  echo "Bucket already exists: gs://${BUCKET}"
fi


cat <<'EOF' > lifecycle.json
{
  "rule": [{
    "action": {"type": "Delete"},
    "condition": {"age": 90, "matchesPrefix": ["pipeline_root/"]}
  }]
}
EOF

gcloud storage buckets update "gs://${BUCKET}" --lifecycle-file=lifecycle.json
rm lifecycle.json

echo "Lifecycle policy set (pipeline_root/ auto-deleted after 90 days)."
echo ""
echo "=== 3. Creating Service Account ==="
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="Vertex AI MLOps Service Account" \
    --project="${PROJECT_ID}"
  echo "Service account created: ${SA_EMAIL}"
else
  echo "Service account already exists: ${SA_EMAIL}"
fi

echo ""
echo "=== 4. Granting IAM roles ==="
ROLES=(
  "roles/aiplatform.user"
  "roles/bigquery.dataEditor"
  "roles/bigquery.jobUser"
  "roles/storage.objectAdmin"
  "roles/iam.serviceAccountUser"
  "roles/cloudfunctions.invoker"
  "roles/logging.logWriter"
  "roles/monitoring.metricWriter"
  "roles/artifactregistry.reader"
)
for ROLE in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet
  echo "  Granted: ${ROLE}"
done

echo ""
echo "=== 5. Initializing Vertex AI Experiment ==="
PROJECT_ID="${PROJECT_ID}" REGION="${REGION}" python3 - <<'PYEOF'
import os, google.cloud.aiplatform as aip
project = os.environ["PROJECT_ID"]
region  = os.environ["REGION"]
aip.init(project=project, location=region)
try:
    exp = aip.Experiment.create(
        experiment_name="used-car-price-prediction",
        description="XGBoost regression for used car price prediction",
        project=project,
        location=region,
    )
    print("  Experiment created:", exp.resource_name)
except Exception as exc:
    if "already exists" in str(exc).lower():
        print("  Experiment already exists — skipping.")
    else:
        raise
PYEOF

echo ""
echo "=== 6. Creating BigQuery dataset ==="
if ! bq ls --project_id="${PROJECT_ID}" "${BQ_DATASET}" &>/dev/null; then
  bq mk --dataset --location=US "${PROJECT_ID}:${BQ_DATASET}"
  echo "BigQuery dataset created: ${BQ_DATASET}"
else
  echo "BigQuery dataset already exists: ${BQ_DATASET}"
fi

echo ""
echo "=== 7. Setting up Workload Identity Federation (GitHub Actions CI/CD) ==="
echo "NOTE: Replace GITHUB_ORG and GITHUB_REPO with your actual values."
echo ""
echo "Run the following commands to complete WIF setup:"
cat << 'WIF_INSTRUCTIONS'
# --- Workload Identity Pool ---
gcloud iam workload-identity-pools create "github-actions-pool" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# --- OIDC Provider ---
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --display-name="GitHub OIDC Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# --- Bind SA to GitHub repo ---
GITHUB_ORG="<YOUR_GITHUB_ORG>"
GITHUB_REPO="<YOUR_GITHUB_REPO>"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')

gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-actions-pool/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}"

# Get the WIF provider resource name (add to GitHub Actions secret WIF_PROVIDER):
gcloud iam workload-identity-pools providers describe "github-provider" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --format="value(name)"
WIF_INSTRUCTIONS

echo ""
echo "============================================================"
echo " Setup complete! Next steps:"
echo ""
echo "  1. Generate and upload data:"
echo "     python data/generate_and_upload_data.py --project ${PROJECT_ID}"
echo ""
echo "  2. Compile pipelines:"
echo "     python scripts/compile_pipelines.py --upload \\"
echo "       --gcs-prefix gs://${BUCKET}/compiled_pipelines"
echo ""
echo "  3. Run training pipeline:"
echo "     python scripts/run_training_pipeline.py \\"
echo "       --project ${PROJECT_ID} \\"
echo "       --bucket gs://${BUCKET}"
echo ""
echo "  4. Deploy Cloud Function trigger:"
echo "     bash triggers/deploy_trigger.sh"
echo ""
echo "  5. Push to GitHub to trigger CI/CD pipeline"
echo "============================================================"
import os
import uuid
import functions_framework
from datetime import datetime


@functions_framework.http
def trigger_inference_pipeline(request):
    """HTTP-triggered Cloud Function entry point (also works with Cloud Scheduler)."""
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    PROJECT_ID           = os.environ["PROJECT_ID"]
    REGION               = os.environ.get("REGION", "us-central1")
    STAGING_BUCKET       = os.environ["STAGING_BUCKET"]
    BQ_DATASET           = os.environ.get("BQ_DATASET", "used_car_mlops")
    INFERENCE_TABLE      = os.environ.get("INFERENCE_TABLE", "car_prices_inference")
    PROCESSED_TABLE      = os.environ.get("PROCESSED_TABLE", "car_prices_inference_processed")
    PREDICTIONS_TABLE    = os.environ.get("PREDICTIONS_TABLE", "car_predictions")
    COMPILED_YAML_GCS    = os.environ["COMPILED_INFERENCE_YAML_GCS"]
    ENDPOINT_ID          = os.environ.get("PRODUCTION_ENDPOINT_ID", "")
    MODEL_RESOURCE_NAME  = os.environ.get("MODEL_RESOURCE_NAME", "")
    PREPROCESSOR_GCS     = os.environ.get("PREPROCESSOR_GCS_PATH", "")
    BATCH_THRESHOLD      = int(os.environ.get("BATCH_THRESHOLD", "100"))

    log.info(f"Trigger fired at {datetime.utcnow().isoformat()}")

    # ── Check for new rows ─────────────────────────────────────────────────────
    from google.cloud import bigquery
    bq = bigquery.Client(project=PROJECT_ID)
    count_query = f"""
        SELECT COUNT(*) AS cnt
        FROM `{PROJECT_ID}.{BQ_DATASET}.{INFERENCE_TABLE}` inf
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.{PROCESSED_TABLE}` proc
          USING (row_id)
        WHERE proc.row_id IS NULL
    """
    result = list(bq.query(count_query).result())
    new_row_count = result[0].cnt
    log.info(f"New unprocessed rows: {new_row_count}")

    if new_row_count == 0:
        msg = "No new rows — skipping pipeline submission."
        log.info(msg)
        return (msg, 200)

    if not MODEL_RESOURCE_NAME or not ENDPOINT_ID:
        msg = "MODEL_RESOURCE_NAME or PRODUCTION_ENDPOINT_ID not configured — skipping."
        log.warning(msg)
        return (msg, 412)

    # ── Submit Vertex AI PipelineJob ───────────────────────────────────────────
    # aip.PipelineJob accepts GCS URIs directly — no local download needed.
    import google.cloud.aiplatform as aip
    aip.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

    run_id = f"inf-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    pipeline_params = {
        "project_id":              PROJECT_ID,
        "region":                  REGION,
        "staging_bucket":          STAGING_BUCKET,
        "bq_dataset":              BQ_DATASET,
        "inference_table":         INFERENCE_TABLE,
        "processed_table":         PROCESSED_TABLE,
        "predictions_table":       PREDICTIONS_TABLE,
        "model_resource_name":     MODEL_RESOURCE_NAME,
        "endpoint_id":             ENDPOINT_ID,
        "preprocessor_gcs_path":   PREPROCESSOR_GCS,
        "batch_threshold":         BATCH_THRESHOLD,
        "pipeline_run_id":         run_id,
    }

    pipeline_job = aip.PipelineJob(
        display_name=f"inference-{run_id}",
        template_path=COMPILED_YAML_GCS,          # GCS URI passed directly
        pipeline_root=f"{STAGING_BUCKET}/pipeline_root/inference",
        parameter_values=pipeline_params,
        enable_caching=False,   # Always re-run inference on fresh data
    )
    sa_email = f"vertex-mlops-sa@{PROJECT_ID}.iam.gserviceaccount.com"
    pipeline_job.submit(service_account=sa_email)

    log.info(f"Inference pipeline submitted: {pipeline_job.resource_name}")
    log.info(f"  new_rows={new_row_count}, run_id={run_id}")

    return ({
        "status": "submitted",
        "run_id": run_id,
        "new_rows": new_row_count,
        "pipeline_resource_name": pipeline_job.resource_name,
    }, 200)
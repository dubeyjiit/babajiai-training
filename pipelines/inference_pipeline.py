"""
Vertex AI Inference Pipeline  —  Used Car Price Prediction
==========================================================
Triggered whenever new rows arrive in BigQuery `car_prices_inference` table
(via Cloud Function + Cloud Scheduler polling).

Two modes based on row count:
  < batch_threshold rows  → Online prediction via Vertex AI Endpoint
  >= batch_threshold rows → Vertex AI Batch Prediction Job

Stages:
  1. check_new_data       — Poll BQ for unprocessed inference rows
  2. preprocess_inference — Apply the same ColumnTransformer (fetched from GCS)
  3. run_inference         — Online OR batch prediction
  4. store_predictions     — Write results back to BQ `car_predictions`
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kfp import dsl
from kfp.dsl import pipeline, component, Output, Input, Dataset, Metrics
from typing import NamedTuple


# ── Component: Check for new inference data ────────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-bigquery[pandas]==3.25.0",
        "pandas==2.2.2",
        "db-dtypes==1.3.0",
        "pyarrow==15.0.2",
    ],
)
def check_new_data_op(
    project_id: str,
    bq_dataset: str,
    inference_table: str,
    processed_table: str,
    batch_threshold: int,
    new_data: Output[Dataset],
    data_check: Output[Metrics],
) -> NamedTuple("DataCheckOutputs", [("row_count", int), ("inference_mode", str)]):
    """Query BQ for rows not yet in the processed-tracking table."""
    from google.cloud import bigquery
    from collections import namedtuple

    client = bigquery.Client(project=project_id)

    query = f"""
        SELECT inf.*
        FROM `{project_id}.{bq_dataset}.{inference_table}` inf
        LEFT JOIN `{project_id}.{bq_dataset}.{processed_table}` proc
          USING (row_id)
        WHERE proc.row_id IS NULL
        ORDER BY inf.row_id
    """
    print(f"=== Checking for new inference rows ===\n{query}")

    df = client.query(query).to_dataframe()
    row_count = len(df)
    print(f"Found {row_count} unprocessed rows")

    df.to_csv(new_data.path + ".csv", index=False)

    inference_mode = "batch" if row_count >= batch_threshold else "online"
    print(f"Inference mode: {inference_mode} (threshold={batch_threshold})")

    data_check.log_metric("new_row_count", row_count)
    data_check.log_metric("batch_threshold", batch_threshold)
    data_check.log_metric("inference_mode_is_batch", int(inference_mode == "batch"))

    DataCheckOutputs = namedtuple("DataCheckOutputs", ["row_count", "inference_mode"])
    return DataCheckOutputs(row_count=row_count, inference_mode=inference_mode)


# ── Component: Preprocess inference data ──────────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "scikit-learn==1.4.2",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "joblib==1.4.2",
        "google-cloud-storage==2.17.0",
    ],
)
def preprocess_inference_op(
    project_id: str,
    staging_bucket: str,
    preprocessor_gcs_path: str,           # e.g. gs://bucket/models/latest/preprocessor.pkl
    new_data: Input[Dataset],
    preprocessed_inference: Output[Dataset],
) -> None:
    """
    Apply feature engineering and column selection ONLY.

    The saved artifact is a full sklearn Pipeline(preprocessor + model).
    When we call pipeline.predict(X), the Pipeline applies ColumnTransformer
    internally. So this component must output RAW+ENGINEERED features
    (before scaling/encoding), not already-transformed ones.

    Feeding pre-scaled features into the Pipeline would double-preprocess
    and produce garbage predictions.
    """
    import io
    import joblib
    import pandas as pd
    from google.cloud import storage

    print("=== Feature engineering for inference ===")
    df = pd.read_csv(new_data.path + ".csv")
    print(f"Input rows: {len(df)}")

    # Download preprocessor metadata to get the exact column order used at training time
    bucket_name = preprocessor_gcs_path.replace("gs://", "").split("/")[0]
    blob_path   = "/".join(preprocessor_gcs_path.replace("gs://", "").split("/")[1:])
    client = storage.Client(project=project_id)
    blob = client.bucket(bucket_name).blob(blob_path)
    meta = joblib.load(io.BytesIO(blob.download_as_bytes()))["meta"]

    NUMERIC_FEATURES     = meta["numeric_features"]
    CATEGORICAL_FEATURES = meta["categorical_features"]

    # Feature engineering — identical to training preprocessing
    df["car_age"]          = 2024 - df["year"]
    df["mileage_per_year"] = df["mileage"] / (df["car_age"].replace(0, 1))
    df["hp_per_liter"]     = df["horsepower"] / df["engine_size"]

    # Select columns in the exact order the Pipeline was fitted on
    # Do NOT call preprocessor.transform() here — the full Pipeline does that internally
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    out_df = X.copy()
    if "row_id" in df.columns:
        out_df.insert(0, "row_id", df["row_id"].values)

    out_df.to_csv(preprocessed_inference.path + ".csv", index=False)
    print(f"Feature-engineered rows saved: {len(out_df)}")
    print(f"Columns (raw+engineered, pre-transform): {list(X.columns)}")


# ── Component: Online prediction via Vertex AI Endpoint ───────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.60.0",
        "pandas==2.2.2",
        "numpy==1.26.4",
    ],
)
def online_prediction_op(
    project_id: str,
    region: str,
    endpoint_id: str,
    preprocessed_inference: Input[Dataset],
    predictions: Output[Dataset],
    prediction_metrics: Output[Metrics],
) -> None:
    """
    Send rows to the Vertex AI online endpoint in mini-batches.

    The input CSV contains RAW+ENGINEERED features (not yet scaled/encoded).
    The endpoint serves the full sklearn Pipeline (preprocessor + model),
    so it handles ColumnTransformer internally — no pre-scaling needed here.
    """
    import pandas as pd
    import google.cloud.aiplatform as aip

    aip.init(project=project_id, location=region)

    df = pd.read_csv(preprocessed_inference.path + ".csv")
    row_ids = df["row_id"].tolist() if "row_id" in df.columns else [str(i) for i in range(len(df))]
    # Drop row_id — endpoint expects only feature columns
    feature_df = df.drop(columns=["row_id"], errors="ignore")

    endpoint = aip.Endpoint(endpoint_name=endpoint_id)

    BATCH_SIZE = 50   # Vertex AI online prediction: max instances per request
    all_preds = []
    for start in range(0, len(feature_df), BATCH_SIZE):
        batch = feature_df.iloc[start:start + BATCH_SIZE]
        instances = batch.values.tolist()   # list of feature arrays (raw+engineered)
        response = endpoint.predict(instances=instances)
        all_preds.extend(response.predictions)

    pred_df = pd.DataFrame({"row_id": row_ids, "predicted_price": all_preds})
    pred_df["predicted_price"] = pred_df["predicted_price"].round(2)
    pred_df.to_csv(predictions.path + ".csv", index=False)

    prediction_metrics.log_metric("rows_predicted", len(pred_df))
    prediction_metrics.log_metric("mean_predicted_price", float(pd.Series(all_preds).mean()))
    print(f"Online predictions complete: {len(pred_df)} rows")


# ── Component: Batch scoring via local model inference ────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "scikit-learn==1.4.2",
        "xgboost==2.0.3",
        "google-cloud-aiplatform==1.60.0",
        "google-cloud-storage==2.17.0",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "joblib==1.4.2",
    ],
)
def batch_prediction_op(
    project_id: str,
    region: str,
    staging_bucket: str,
    model_resource_name: str,
    batch_machine_type: str,           # kept for pipeline signature compatibility
    batch_starting_replicas: int,      # kept for pipeline signature compatibility
    batch_max_replicas: int,           # kept for pipeline signature compatibility
    preprocessed_inference: Input[Dataset],
    predictions: Output[Dataset],
    prediction_metrics: Output[Metrics],
) -> None:
    """
    Score inference rows locally by loading the model artifact from GCS.

    Why local scoring instead of Vertex AI Batch Prediction service:
      The sklearn-cpu.1-3 pre-built serving container is incompatible with
      a pickled sklearn Pipeline that contains XGBoost 2.0.3 — it raises
      "'numpy.ndarray' object has no attribute 'predict'" on every request.
      Loading the model directly with the same sklearn==1.4.2 / xgboost==2.0.3
      versions used during training is reliable and avoids VM spin-up latency.
    """
    import io
    import joblib
    import numpy as np
    import pandas as pd
    import google.cloud.aiplatform as aip
    from google.cloud import storage

    aip.init(project=project_id, location=region)

    # ── Load preprocessed features ─────────────────────────────────────────────
    df = pd.read_csv(preprocessed_inference.path + ".csv")
    row_ids    = df["row_id"].tolist() if "row_id" in df.columns else list(range(len(df)))
    feature_df = df.drop(columns=["row_id"], errors="ignore")
    print(f"Scoring {len(feature_df)} rows · {feature_df.shape[1]} features")

    # ── Fetch model artifact URI from Vertex AI Model Registry ────────────────
    model_obj    = aip.Model(model_resource_name)
    artifact_uri = model_obj.uri                   # gs://.../models/{run-id}
    gcs_pkl      = artifact_uri.rstrip("/") + "/model.pkl"
    print(f"Loading model from: {gcs_pkl}")

    # ── Download and deserialise full sklearn Pipeline ─────────────────────────
    gcs_client  = storage.Client(project=project_id)
    bucket_name = gcs_pkl.replace("gs://", "").split("/")[0]
    blob_path   = "/".join(gcs_pkl.replace("gs://", "").split("/")[1:])
    pipeline    = joblib.load(io.BytesIO(
        gcs_client.bucket(bucket_name).blob(blob_path).download_as_bytes()
    ))
    print(f"Model type: {type(pipeline)}")

    # ── Local prediction — Pipeline handles ColumnTransformer internally ───────
    # Pass DataFrame (not .values) so ColumnTransformer can index by column name
    y_pred = pipeline.predict(feature_df)
    print(f"Predictions: min={y_pred.min():.2f}  max={y_pred.max():.2f}  mean={y_pred.mean():.2f}")

    # ── Save results ───────────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "row_id":          row_ids,
        "predicted_price": y_pred.round(2).tolist(),
    })
    pred_df.to_csv(predictions.path + ".csv", index=False)

    prediction_metrics.log_metric("rows_predicted",        len(pred_df))
    prediction_metrics.log_metric("mean_predicted_price",  float(np.mean(y_pred)))
    prediction_metrics.log_metric("min_predicted_price",   float(np.min(y_pred)))
    prediction_metrics.log_metric("max_predicted_price",   float(np.max(y_pred)))
    print(f"Local batch scoring complete: {len(pred_df)} rows")


# ── Component: Store predictions in BigQuery ───────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-bigquery[pandas]==3.25.0",
        "pandas==2.2.2",
        "pyarrow==15.0.2",
        "db-dtypes==1.3.0",
    ],
)
def store_predictions_op(
    project_id: str,
    bq_dataset: str,
    predictions_table: str,
    processed_table: str,
    model_resource_name: str,
    pipeline_run_id: str,
    predictions: Input[Dataset],
    store_metrics: Output[Metrics],
) -> None:
    """Write predictions to BQ and mark source rows as processed."""
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    pred_df = pd.read_csv(predictions.path + ".csv")

    # Use pd.Timestamp so pyarrow can convert it to TIMESTAMP natively.
    # datetime.isoformat() returns a plain string which pyarrow cannot cast.
    now_ts = pd.Timestamp.utcnow()
    pred_df["model_version"]   = model_resource_name.split("/models/")[-1]
    pred_df["pipeline_run_id"] = pipeline_run_id
    pred_df["prediction_ts"]   = now_ts

    # Append predictions
    table_ref = f"{project_id}.{bq_dataset}.{predictions_table}"
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("row_id",          "STRING"),
            bigquery.SchemaField("predicted_price",  "FLOAT64"),
            bigquery.SchemaField("model_version",    "STRING"),
            bigquery.SchemaField("pipeline_run_id",  "STRING"),
            bigquery.SchemaField("prediction_ts",    "TIMESTAMP"),
        ],
    )
    job = client.load_table_from_dataframe(
        pred_df[["row_id", "predicted_price", "model_version", "pipeline_run_id", "prediction_ts"]],
        table_ref,
        job_config=job_config,
    )
    job.result()
    print(f"Wrote {len(pred_df)} predictions → {table_ref}")

    # Mark rows as processed (prevents re-processing on next run)
    processed_ref = f"{project_id}.{bq_dataset}.{processed_table}"
    proc_df = pd.DataFrame({
        "row_id":       pred_df["row_id"],
        "processed_ts": now_ts,
    })
    proc_job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("row_id",        "STRING"),
            bigquery.SchemaField("processed_ts",  "TIMESTAMP"),
        ],
    )
    proc_job = client.load_table_from_dataframe(proc_df, processed_ref, job_config=proc_job_config)
    proc_job.result()
    print(f"Marked {len(proc_df)} rows as processed → {processed_ref}")

    store_metrics.log_metric("predictions_stored", len(pred_df))
    store_metrics.log_metric("prediction_table", table_ref)


# ── Pipeline definition ────────────────────────────────────────────────────────
@pipeline(
    name="used-car-inference-pipeline",
    description=(
        "Inference pipeline for used car price prediction. "
        "Triggered by new rows in BigQuery. "
        "Routes to online or batch prediction based on row count."
    ),
)
def inference_pipeline(
    project_id: str,
    region: str = "us-central1",
    staging_bucket: str = "gs://<YOUR_PROJECT>-mlops-artifacts",
    # BigQuery
    bq_dataset: str = "used_car_mlops",
    inference_table: str = "car_prices_inference",
    processed_table: str = "car_prices_inference_processed",
    predictions_table: str = "car_predictions",
    # Model & serving
    model_resource_name: str = "",       # Vertex AI model resource name
    endpoint_id: str = "",               # Production endpoint resource name
    preprocessor_gcs_path: str = "",     # GCS path to fitted preprocessor.pkl
    # Batch prediction config
    batch_threshold: int = 100,
    batch_machine_type: str = "n1-standard-4",
    batch_starting_replicas: int = 2,
    batch_max_replicas: int = 10,
    # Run ID
    pipeline_run_id: str = "manual",
) -> None:

    # ── 1. Check for new data ──────────────────────────────────────────────────
    check = check_new_data_op(
        project_id=project_id,
        bq_dataset=bq_dataset,
        inference_table=inference_table,
        processed_table=processed_table,
        batch_threshold=batch_threshold,
    )
    check.set_display_name("1 · Check New BQ Data")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    with dsl.If(check.outputs["row_count"] > 0, name="Has new rows"):

        preprocess = preprocess_inference_op(
            project_id=project_id,
            staging_bucket=staging_bucket,
            preprocessor_gcs_path=preprocessor_gcs_path,
            new_data=check.outputs["new_data"],
        )
        preprocess.set_display_name("2 · Preprocess Inference Data")

        # ── 3a. Online prediction ───────────────────────────────────────────
        with dsl.If(check.outputs["inference_mode"] == "online", name="Online prediction"):
            online_pred = online_prediction_op(
                project_id=project_id,
                region=region,
                endpoint_id=endpoint_id,
                preprocessed_inference=preprocess.outputs["preprocessed_inference"],
            )
            online_pred.set_display_name("3a · Online Endpoint Prediction")

            store_online = store_predictions_op(
                project_id=project_id,
                bq_dataset=bq_dataset,
                predictions_table=predictions_table,
                processed_table=processed_table,
                model_resource_name=model_resource_name,
                pipeline_run_id=pipeline_run_id,
                predictions=online_pred.outputs["predictions"],
            )
            store_online.set_display_name("4a · Store Online Predictions → BQ")

        # ── 3b. Batch prediction ────────────────────────────────────────────
        with dsl.If(check.outputs["inference_mode"] == "batch", name="Batch prediction"):
            batch_pred = batch_prediction_op(
                project_id=project_id,
                region=region,
                staging_bucket=staging_bucket,
                model_resource_name=model_resource_name,
                batch_machine_type=batch_machine_type,
                batch_starting_replicas=batch_starting_replicas,
                batch_max_replicas=batch_max_replicas,
                preprocessed_inference=preprocess.outputs["preprocessed_inference"],
            )
            batch_pred.set_display_name("3b · Batch Prediction Job (Vertex AI Batch Inference)")

            store_batch = store_predictions_op(
                project_id=project_id,
                bq_dataset=bq_dataset,
                predictions_table=predictions_table,
                processed_table=processed_table,
                model_resource_name=model_resource_name,
                pipeline_run_id=pipeline_run_id,
                predictions=batch_pred.outputs["predictions"],
            )
            store_batch.set_display_name("4b · Store Batch Predictions → BQ")


def build_inference_pipeline():
    return inference_pipeline

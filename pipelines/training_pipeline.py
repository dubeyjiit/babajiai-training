"""
Vertex AI Training Pipeline  —  Used Car Price Prediction
=========================================================
Stages:
   1. data_ingestion      — BQ → GCS, register Vertex AI Dataset
   2. data_preprocessing  — Feature engineering + ColumnTransformer fit
   3. model_training      — XGBoost, logs to Vertex AI Experiments
   4. model_evaluation    — Test-set metrics, deployment gate
   5. model_registry      — Upload to Vertex AI Model Registry (versioned)
   6. model_serving_stag  — Deploy to Staging endpoint
   7. model_serving_prod  — (conditional) Promote to Production endpoint

The compiled YAML is uploaded to GCS and submitted as a PipelineJob,
which makes it visible in Agent Platform → Pipelines.
"""


from kfp import dsl
from kfp.dsl import pipeline

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from components.data_ingestion    import data_ingestion_op
from components.data_preprocessing import data_preprocessing_op
from components.model_training    import model_training_op
from components.model_evaluation  import model_evaluation_op
from components.model_registry    import model_registry_op
from components.model_serving     import model_serving_op


@pipeline(
    name="used-car-training-pipeline",
    description=(
        "End-to-end MLOps training pipeline for used car price prediction. "
        "Ingests from BigQuery, preprocesses, trains XGBoost, evaluates, "
        "registers in Model Registry, and deploys to Staging + Production endpoints."
    ),
)
def training_pipeline(
    # ── GCP project ───────────────────────────────────────────────────────────
    project_id: str,
    region: str = "us-central1",
    staging_bucket: str = "gs://<YOUR_PROJECT>-mlops-artifacts",
    # ── BigQuery ──────────────────────────────────────────────────────────────
    bq_dataset: str = "used_car_mlops",
    bq_train_table: str = "car_prices_train",
    bq_test_table: str = "car_prices_test",
    # ── Vertex AI identifiers ─────────────────────────────────────────────────
    vertex_dataset_name: str = "used-car-prices-dataset",
    experiment_name: str = "used-car-price-prediction",
    model_name: str = "used-car-price-xgb",
    serving_container_image: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    # ── Run-level identifiers (injected by CI or caller) ─────────────────────
    pipeline_run_id: str = "manual",
    # ── Hyperparameters ───────────────────────────────────────────────────────
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 3,
    # ── Evaluation thresholds ─────────────────────────────────────────────────
    rmse_threshold: float = 4500.0,
    r2_threshold: float = 0.80,
    # ── Serving config ────────────────────────────────────────────────────────
    serving_machine_type: str = "n1-standard-4",
    staging_min_replicas: int = 1,
    staging_max_replicas: int = 2,
    prod_min_replicas: int = 2,
    prod_max_replicas: int = 5,
    # ── Feature flag: promote to production after staging ─────────────────────
    promote_to_production: bool = True,
) -> None:

    # Stable run name so every component inside the same pipeline run
    # shares the same Vertex AI Experiment run entry.
    experiment_run_name = f"run-{pipeline_run_id}"

    # ── 1. Data Ingestion ──────────────────────────────────────────────────────
    ingestion = data_ingestion_op(
        project_id=project_id,
        region=region,
        bq_dataset=bq_dataset,
        bq_train_table=bq_train_table,
        bq_test_table=bq_test_table,
        vertex_dataset_name=vertex_dataset_name,
        staging_bucket=staging_bucket,
    )
    ingestion.set_display_name("1 · Data Ingestion (BigQuery → GCS)")
    ingestion.set_cpu_limit("2").set_memory_limit("8G")

    # ── 2. Data Preprocessing ─────────────────────────────────────────────────
    preprocessing = data_preprocessing_op(
        project_id=project_id,
        region=region,
        experiment_name=experiment_name,
        experiment_run_name=experiment_run_name,
        staging_bucket=staging_bucket,
        train_data=ingestion.outputs["train_data"],
        test_data=ingestion.outputs["test_data"],
    )
    preprocessing.set_display_name("2 · Feature Engineering & Preprocessing")
    preprocessing.set_cpu_limit("4").set_memory_limit("16G")

    # ── 3. Model Training ─────────────────────────────────────────────────────
    training = model_training_op(
        project_id=project_id,
        region=region,
        staging_bucket=staging_bucket,
        experiment_name=experiment_name,
        experiment_run_name=experiment_run_name,
        preprocessed_train=preprocessing.outputs["preprocessed_train"],
        preprocessor_artifact=preprocessing.outputs["preprocessor_artifact"],
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
    )
    training.set_display_name("3 · XGBoost Training (logged to Experiments)")
    training.set_cpu_limit("8").set_memory_limit("32G")

    # ── 4. Model Evaluation ───────────────────────────────────────────────────
    evaluation = model_evaluation_op(
        project_id=project_id,
        region=region,
        staging_bucket=staging_bucket,
        experiment_name=experiment_name,
        experiment_run_name=experiment_run_name,
        rmse_threshold=rmse_threshold,
        r2_threshold=r2_threshold,
        trained_model=training.outputs["trained_model"],
        preprocessed_test=preprocessing.outputs["preprocessed_test"],
        feature_importance_artifact=training.outputs["feature_importance"],
    )
    evaluation.set_display_name("4 · Evaluation & Deployment Gate")

    # ── 5. Model Registry ─────────────────────────────────────────────────────
    # Only runs if evaluation passes the gate
    with dsl.If(evaluation.outputs["deploy_decision"] == "deploy", name="Gate: RMSE+R2 passed"):

        registry = model_registry_op(
            project_id=project_id,
            region=region,
            staging_bucket=staging_bucket,
            model_name=model_name,
            serving_container_image=serving_container_image,
            experiment_name=experiment_name,
            experiment_run_name=experiment_run_name,
            pipeline_run_id=pipeline_run_id,
            test_rmse=evaluation.outputs["test_rmse"],
            test_r2=evaluation.outputs["test_r2"],
            trained_model=training.outputs["trained_model"],
        )
        registry.set_display_name("5 · Model Registry (versioned)")

        # ── 6. Deploy to Staging ──────────────────────────────────────────────
        serving_staging = model_serving_op(
            project_id=project_id,
            region=region,
            staging_bucket=staging_bucket,
            environment="staging",
            machine_type=serving_machine_type,
            min_replicas=staging_min_replicas,
            max_replicas=staging_max_replicas,
            model_resource_name=registry.outputs["model_resource_name"],
            registered_model_info=registry.outputs["registered_model_info"],
        )
        serving_staging.set_display_name("6 · Deploy → Staging Endpoint")

        # ── 7. Deploy to Production (optional) ───────────────────────────────
        with dsl.If(promote_to_production == True, name="Promote to Production"):  # noqa: E712
            serving_prod = model_serving_op(
                project_id=project_id,
                region=region,
                staging_bucket=staging_bucket,
                environment="production",
                machine_type=serving_machine_type,
                min_replicas=prod_min_replicas,
                max_replicas=prod_max_replicas,
                model_resource_name=registry.outputs["model_resource_name"],
                registered_model_info=registry.outputs["registered_model_info"],
            )
            serving_prod.set_display_name("7 · Deploy → Production Endpoint")
            serving_prod.after(serving_staging)


def build_training_pipeline() -> None:
    """Return the pipeline function (used by compile_pipelines.py)."""
    return training_pipeline

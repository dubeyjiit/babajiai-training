from kfp.dsl import component, Input, Output, Model, Artifact
from typing import NamedTuple


@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.60.0",
        "google-cloud-storage==2.17.0",
        "scikit-learn==1.4.2",
        "xgboost==2.0.3",
        "joblib==1.4.2",
    ],
)
def model_registry_op(
    project_id: str,
    region: str,
    staging_bucket: str,
    model_name: str,
    serving_container_image: str,
    experiment_name: str,
    experiment_run_name: str,
    pipeline_run_id: str,
    test_rmse: float,
    test_r2: float,
    trained_model: Input[Model],
    registered_model_info: Output[Artifact],
) -> NamedTuple("RegistryOutputs", [("model_resource_name", str), ("model_version_id", str)]):
    """Upload model artifact to GCS, register in Vertex AI Model Registry with metadata."""
    import json
    import uuid
    import google.cloud.aiplatform as aip
    from google.cloud import storage
    from collections import namedtuple

    aip.init(project=project_id, location=region, staging_bucket=staging_bucket,
             experiment=experiment_name)

    # ── Copy model artifact to GCS ─────────────────────────────────────────────
    # Vertex AI expects model.pkl at gs://<bucket>/model/model.pkl
    bucket_name = staging_bucket.replace("gs://", "").split("/")[0]
    gcs_model_dir = f"models/{experiment_run_name}"
    gcs_client = storage.Client(project=project_id)
    bucket = gcs_client.bucket(bucket_name)

    local_model_path = trained_model.path + ".pkl"
    gcs_model_blob = f"{gcs_model_dir}/model.pkl"
    bucket.blob(gcs_model_blob).upload_from_filename(local_model_path)
    gcs_model_uri = f"gs://{bucket_name}/{gcs_model_dir}"
    print(f"Model artifact uploaded → {gcs_model_uri}")

    # ── Register model / create new version ───────────────────────────────────
    # Check if a model with this name already exists
    existing_models = aip.Model.list(
        filter=f'displayName="{model_name}"',
        project=project_id,
        location=region,
    )

    labels = {
        "experiment":    experiment_name[:63].replace(" ", "-").lower(),
        "pipeline_run":  pipeline_run_id[:63].replace(" ", "-").lower()[:63],
        "framework":     "xgboost",
        "task":          "regression",
    }

    model_description = (
        f"Used car price prediction — XGBoost. "
        f"Run: {experiment_run_name} | RMSE: {test_rmse:.0f} | R²: {test_r2:.4f}"
    )

    # Pre-built sklearn containers already define /predict + /ping routes internally.
    # Specifying serving_container_health_route here overrides to a wrong path and
    # causes health checks to fail during endpoint.deploy(). Leave routes unset.
    if existing_models:
        # Upload as a new version of the existing model
        parent_model = existing_models[0].resource_name
        print(f"Registering new version under existing model: {parent_model}")
        model = aip.Model.upload(
            display_name=model_name,
            artifact_uri=gcs_model_uri,
            serving_container_image_uri=serving_container_image,
            description=model_description,
            labels=labels,
            parent_model=parent_model,          # ← creates a new version
            is_default_version=True,
        )
    else:
        print(f"No existing model found. Registering as new model: {model_name}")
        model = aip.Model.upload(
            display_name=model_name,
            artifact_uri=gcs_model_uri,
            serving_container_image_uri=serving_container_image,
            description=model_description,
            labels=labels,
        )

    model_resource_name = model.resource_name
    # Version ID is the last segment of the version resource name
    # resource_name format: projects/.../models/<model_id>
    model_id = model_resource_name.split("/models/")[-1]
    # Fetch version details
    version_id = getattr(model, "version_id", "1")

    print(f"Model registered: {model_resource_name}")
    print(f"Version ID:       {version_id}")

    # ── Write ML Metadata (visible in Agent Platform → Metadata) ──────────────
    info = {
        "model_resource_name":  model_resource_name,
        "model_id":             model_id,
        "version_id":           version_id,
        "gcs_artifact_uri":     gcs_model_uri,
        "display_name":         model_name,
        "experiment_run_name":  experiment_run_name,
        "test_rmse":            test_rmse,
        "test_r2":              test_r2,
        "serving_container":    serving_container_image,
    }
    with open(registered_model_info.path + ".json", "w") as f:
        json.dump(info, f, indent=2)

    # Log lineage via Vertex AI Experiment
    unique_run_name = f"{experiment_run_name}-reg-{uuid.uuid4().hex[:6]}"
    print(f"Experiment run name: {unique_run_name}")
    with aip.start_run(unique_run_name):
        aip.log_params({
            "model_resource_name": model_resource_name,
            "model_version_id":    version_id,
            "gcs_artifact_uri":    gcs_model_uri,
        })

    print("=== Model Registry complete ===")
    RegistryOutputs = namedtuple("RegistryOutputs", ["model_resource_name", "model_version_id"])
    return RegistryOutputs(model_resource_name=model_resource_name, model_version_id=str(version_id))

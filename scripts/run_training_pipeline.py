import argparse
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import google.cloud.aiplatform as aip


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def submit_training_pipeline(
    project_id: str,
    region: str,
    staging_bucket: str,
    run_id: str,
    promote_to_production: bool = True,
    compiled_yaml: str | None = None,
) -> None:
    cfg = load_config()

    aip.init(project=project_id, location=region, staging_bucket=staging_bucket)

    # Use compiled YAML if provided, otherwise compile on the fly
    if compiled_yaml and os.path.exists(compiled_yaml):
        template_path = compiled_yaml
    else:
        compiled_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "compiled_pipelines")
        template_path = os.path.join(compiled_dir, "training_pipeline.yaml")
        if not os.path.exists(template_path):
            print("Compiled YAML not found. Compiling now...")
            from compile_pipelines import compile_pipelines
            compile_pipelines()

    pipeline_params = {
        "project_id":              project_id,
        "region":                  region,
        "staging_bucket":          staging_bucket,
        "bq_dataset":              cfg["bigquery"]["dataset"],
        "bq_train_table":          cfg["bigquery"]["train_table"],
        "bq_test_table":           cfg["bigquery"]["test_table"],
        "vertex_dataset_name":     cfg["vertex_ai"]["dataset_name"],
        "experiment_name":         cfg["vertex_ai"]["experiment_name"],
        "model_name":              cfg["vertex_ai"]["model_name"],
        "serving_container_image": cfg["vertex_ai"]["serving_container_image"],
        "pipeline_run_id":         run_id,
        "n_estimators":            cfg["training"]["n_estimators"],
        "max_depth":               cfg["training"]["max_depth"],
        "learning_rate":           cfg["training"]["learning_rate"],
        "subsample":               cfg["training"]["subsample"],
        "colsample_bytree":        cfg["training"]["colsample_bytree"],
        "min_child_weight":        cfg["training"]["min_child_weight"],
        "rmse_threshold":          cfg["training"]["rmse_threshold"],
        "r2_threshold":            cfg["training"]["r2_threshold"],
        "serving_machine_type":    cfg["serving"]["machine_type"],
        "staging_min_replicas":    cfg["serving"]["min_replicas_staging"],
        "staging_max_replicas":    cfg["serving"]["max_replicas_staging"],
        "prod_min_replicas":       cfg["serving"]["min_replicas_production"],
        "prod_max_replicas":       cfg["serving"]["max_replicas_production"],
        "promote_to_production":   promote_to_production,
    }

    job_display_name = f"training-{run_id}"
    service_account  = cfg["project"]["service_account"]

    print("=== Submitting training pipeline ===")
    print(f"  Display name:    {job_display_name}")
    print(f"  Run ID:          {run_id}")
    print(f"  Template:        {template_path}")
    print(f"  Service account: {service_account}")
    print(f"  Promote prod:    {promote_to_production}")

    pipeline_job = aip.PipelineJob(
        display_name=job_display_name,
        template_path=template_path,
        pipeline_root=f"{staging_bucket}/pipeline_root",
        parameter_values=pipeline_params,
        enable_caching=True,
    )
    # Pass the dedicated MLOps SA so each component has the right permissions
    pipeline_job.submit(service_account=service_account)
    print("\nPipeline submitted!")
    print(f"  Resource name: {pipeline_job.resource_name}")
    print(f"  Console URL:   https://console.cloud.google.com/vertex-ai/pipelines/runs?project={project_id}&region={region}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",     required=True)
    parser.add_argument("--bucket",      required=True, help="GCS staging bucket (gs://...)")
    parser.add_argument("--region",      default="us-central1")
    parser.add_argument("--run-id",      default=None,  help="Unique run ID (auto-generated if omitted)")
    parser.add_argument("--compiled-yaml", default=None)
    parser.add_argument("--no-promote",  action="store_true", help="Skip production deployment")
    args = parser.parse_args()

    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    submit_training_pipeline(
        project_id=args.project,
        region=args.region,
        staging_bucket=args.bucket,
        run_id=run_id,
        promote_to_production=not args.no_promote,
        compiled_yaml=args.compiled_yaml,)

"""
KFP component: Model Serving

Deploys a registered model to a Vertex AI Endpoint.
Supports two environments:
  - staging:    min 1 replica, used for QA / shadow traffic
  - production: min 2 replicas, receives live traffic

The deployed endpoint is visible in Agent Platform → Endpoints.
Each deploy replaces the previous model version on the same endpoint
(blue-green within the same endpoint resource).
"""

from kfp.dsl import component, Input, Artifact
from typing import NamedTuple


@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.60.0",
        "urllib3<2.0.0",          # kfp dep requires <2; avoids SDK HTTP conflicts
    ],
)
def model_serving_op(
    project_id: str,
    region: str,
    staging_bucket: str,
    environment: str,                     # 'staging' or 'production'
    machine_type: str,
    min_replicas: int,
    max_replicas: int,
    model_resource_name: str,             # Vertex AI model resource name
    registered_model_info: Input[Artifact],
) -> NamedTuple("ServingOutputs", [("endpoint_resource_name", str), ("endpoint_id", str)]):
    """Create or reuse a Vertex AI Endpoint and deploy the model."""
    import traceback
    import google.cloud.aiplatform as aip
    from collections import namedtuple

    aip.init(project=project_id, location=region, staging_bucket=staging_bucket)

    env = environment.lower()
    assert env in ("staging", "production"), f"Unknown environment: {env}"

    endpoint_display_name = f"used-car-{env}-endpoint"
    deployed_model_display_name = f"used-car-xgb-{env}"

    print(f"=== Deploying to {env.upper()} environment ===")
    print(f"  Model resource:  {model_resource_name}")
    print(f"  Endpoint name:   {endpoint_display_name}")
    print(f"  Machine type:    {machine_type}")
    print(f"  Replicas:        {min_replicas}-{max_replicas}")

    # ── Resolve model object ───────────────────────────────────────────────────
    print("  Fetching model object...")
    model = aip.Model(model_resource_name)
    print(f"  Model display name: {model.display_name}")

    # ── Get or create endpoint ─────────────────────────────────────────────────
    print(f"  Looking up endpoint '{endpoint_display_name}'...")
    existing_endpoints = aip.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        project=project_id,
        location=region,
    )

    if existing_endpoints:
        endpoint = existing_endpoints[0]
        print(f"  Reusing existing endpoint: {endpoint.resource_name}")
        deployed = endpoint.list_models()
        for dm in deployed:
            print(f"  Undeploying existing model: {dm.id}")
            try:
                endpoint.undeploy(deployed_model_id=dm.id, sync=True)
            except Exception as e:
                print(f"  WARNING: undeploy failed (non-fatal): {e}")
    else:
        print("  Creating new endpoint...")
        endpoint = aip.Endpoint.create(
            display_name=endpoint_display_name,
            project=project_id,
            location=region,
            labels={"environment": env, "use_case": "used-car-price"},
        )
        print(f"  Created endpoint: {endpoint.resource_name}")

    # ── Deploy model ───────────────────────────────────────────────────────────
    print("  Starting deploy (sync=True, this blocks until healthy)...")
    try:
        endpoint.deploy(
            model=model,
            deployed_model_display_name=deployed_model_display_name,
            machine_type=machine_type,
            min_replica_count=min_replicas,
            max_replica_count=max_replicas,
            traffic_percentage=100,
            sync=True,
        )
    except Exception as deploy_err:
        print("\n!!! endpoint.deploy() FAILED — full traceback below !!!")
        traceback.print_exc()
        raise RuntimeError(
            f"endpoint.deploy() failed for {env} endpoint: {deploy_err}"
        ) from deploy_err

    endpoint_resource_name = endpoint.resource_name
    endpoint_id = endpoint_resource_name.split("/endpoints/")[-1]

    print(f"\n  Endpoint resource name: {endpoint_resource_name}")
    print(f"  Endpoint ID:            {endpoint_id}")
    print(f"=== Model deployed to {env.upper()} ===")

    ServingOutputs = namedtuple("ServingOutputs", ["endpoint_resource_name", "endpoint_id"])
    return ServingOutputs(
        endpoint_resource_name=endpoint_resource_name,
        endpoint_id=endpoint_id,
    )
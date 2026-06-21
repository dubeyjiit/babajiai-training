from kfp.dsl import component, Output, Dataset, Artifact


@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-bigquery[pandas]==3.25.0",
        "google-cloud-aiplatform==1.60.0",
        "google-cloud-storage==2.17.0",
        "pandas==2.2.2",
        "db-dtypes==1.3.0",
        "pyarrow==15.0.2",
    ],
)
def data_ingestion_op(
    project_id: str,
    region: str,
    bq_dataset: str,
    bq_train_table: str,
    bq_test_table: str,
    vertex_dataset_name: str,
    staging_bucket: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    vertex_dataset_resource_name: Output[Artifact],
) -> None:
    """Pull train/test tables from BQ, persist as GCS CSVs, register Vertex Dataset."""
    import json
    import pandas as pd
    from google.cloud import bigquery, storage
    import google.cloud.aiplatform as aip

    aip.init(project=project_id, location=region, staging_bucket=staging_bucket)
    bq_client  = bigquery.Client(project=project_id)
    gcs_client = storage.Client(project=project_id)

    def _bq_to_df(table: str) -> pd.DataFrame:
        query = f"SELECT * FROM `{project_id}.{bq_dataset}.{table}`"
        print(f"  Running: {query}")
        df = bq_client.query(query).to_dataframe()
        print(f"  → {len(df):,} rows, columns: {df.columns.tolist()}")
        return df

    print("=== Fetching training data from BigQuery ===")
    train_df = _bq_to_df(bq_train_table)

    print("=== Fetching test data from BigQuery ===")
    test_df = _bq_to_df(bq_test_table)

    # ── Write to KFP artifact output paths (.csv suffix consistent across all components)
    train_df.to_csv(train_data.path + ".csv", index=False)
    test_df.to_csv(test_data.path   + ".csv", index=False)
    print(f"Train artifact written: {train_data.path}.csv")
    print(f"Test  artifact written: {test_data.path}.csv")

    # ── Also upload a named copy to GCS for the Vertex AI Dataset reference ──
    bucket_name  = staging_bucket.replace("gs://", "").split("/")[0]
    gcs_train_uri = f"{staging_bucket}/data/train/{bq_train_table}.csv"
    gcs_test_uri  = f"{staging_bucket}/data/test/{bq_test_table}.csv"

    def _gcs_blob_path(gcs_uri: str) -> str:
        return "/".join(gcs_uri.replace("gs://", "").split("/")[1:])

    bucket = gcs_client.bucket(bucket_name)
    bucket.blob(_gcs_blob_path(gcs_train_uri)).upload_from_string(
        train_df.to_csv(index=False), content_type="text/csv"
    )
    bucket.blob(_gcs_blob_path(gcs_test_uri)).upload_from_string(
        test_df.to_csv(index=False), content_type="text/csv"
    )
    print(f"Uploaded train CSV → {gcs_train_uri}")
    print(f"Uploaded test  CSV → {gcs_test_uri}")

    # ── Register Vertex AI Tabular Dataset (non-fatal if it fails) ────────────
    # Visible in Agent Platform → Datasets
    print("=== Registering Vertex AI Tabular Dataset ===")
    dataset_resource_name = ""
    try:
        existing = aip.TabularDataset.list(
            filter=f'display_name="{vertex_dataset_name}"',
            project=project_id,
            location=region,
        )
        if existing:
            dataset_resource_name = existing[0].resource_name
            print(f"  Reusing existing dataset: {dataset_resource_name}")
        else:
            ds = aip.TabularDataset.create(
                display_name=vertex_dataset_name,
                gcs_source=gcs_train_uri,
            )
            dataset_resource_name = ds.resource_name
            print(f"  Created dataset: {dataset_resource_name}")
    except Exception as e:
        print(f"  WARNING: Could not register Vertex AI Dataset (non-fatal): {e}")
        dataset_resource_name = "registration-failed"

    # ── Write artifact metadata ───────────────────────────────────────────────
    with open(vertex_dataset_resource_name.path, "w") as f:
        json.dump({
            "resource_name": dataset_resource_name,
            "gcs_train_uri": gcs_train_uri,
            "gcs_test_uri":  gcs_test_uri,
        }, f)

    print("=== Data ingestion complete ===")

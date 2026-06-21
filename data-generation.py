
import argparse
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

SEED = 42
np.random.seed(SEED)

BRANDS = ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Chevrolet", "Hyundai", "Kia", "Nissan", "Audi"]
FUEL_TYPES = ["gasoline", "diesel", "hybrid", "electric"]
TRANSMISSIONS = ["automatic", "manual", "cvt"]
CONDITIONS = ["excellent", "good", "fair", "poor"]

BRAND_PREMIUM = {
    "Toyota": 1.0, "Honda": 0.95, "Ford": 0.85, "BMW": 1.8,
    "Mercedes": 2.0, "Chevrolet": 0.80, "Hyundai": 0.75,
    "Kia": 0.72, "Nissan": 0.88, "Audi": 1.75,
}
FUEL_MULT = {"gasoline": 1.0, "diesel": 1.1, "hybrid": 1.15, "electric": 1.25}
TRANS_MULT = {"automatic": 1.05, "manual": 0.95, "cvt": 1.0}
COND_MULT  = {"excellent": 1.2, "good": 1.0, "fair": 0.75, "poor": 0.50}


def _generate_records(n: int) -> pd.DataFrame:
    year          = np.random.randint(2000, 2024, n)
    mileage       = np.random.uniform(500, 200_000, n).round(0)
    engine_size   = np.random.choice([1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], n)
    horsepower    = np.random.randint(80, 500, n)
    fuel_type     = np.random.choice(FUEL_TYPES, n, p=[0.55, 0.20, 0.15, 0.10])
    transmission  = np.random.choice(TRANSMISSIONS, n, p=[0.65, 0.20, 0.15])
    brand         = np.random.choice(BRANDS, n)
    condition     = np.random.choice(CONDITIONS, n, p=[0.30, 0.40, 0.20, 0.10])
    accident_count = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.50, 0.25, 0.12, 0.07, 0.04, 0.02])
    num_owners    = np.random.choice([1, 2, 3, 4, 5], n, p=[0.40, 0.30, 0.18, 0.08, 0.04])

    base_price = (
        5_000
        + (year - 2000) * 800
        + engine_size * 2_000
        + horsepower * 30
        - mileage * 0.06
        - accident_count * 1_500
        - (num_owners - 1) * 600
    )
    price = (
        base_price
        * np.array([BRAND_PREMIUM[b] for b in brand])
        * np.array([FUEL_MULT[f]     for f in fuel_type])
        * np.array([TRANS_MULT[t]    for t in transmission])
        * np.array([COND_MULT[c]     for c in condition])
        + np.random.normal(0, 1_500, n)
    )
    price = np.clip(price, 500, 120_000).round(2)

    return pd.DataFrame({
        "year":           year.astype(int),
        "mileage":        mileage.astype(int),
        "engine_size":    engine_size,
        "horsepower":     horsepower.astype(int),
        "fuel_type":      fuel_type,
        "transmission":   transmission,
        "brand":          brand,
        "condition":      condition,
        "accident_count": accident_count.astype(int),
        "num_owners":     num_owners.astype(int),
        "price":          price,
    })


def generate_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = _generate_records(10_000)
    full = full.sample(frac=1, random_state=SEED).reset_index(drop=True)

    train = full.iloc[:8_000].copy()
    test  = full.iloc[8_000:].copy()

    inference = _generate_records(500).drop(columns=["price"])
    inference.insert(0, "row_id", [f"inf_{i:05d}" for i in range(len(inference))])

    print(f"Train:     {len(train):,} rows")
    print(f"Test:      {len(test):,} rows")
    print(f"Inference: {len(inference):,} rows")
    print("\nSample train data:")
    print(train.head(3).to_string())
    return train, test, inference


def upload_to_bigquery(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str,
    location: str = "US",
    write_disposition: str = "WRITE_TRUNCATE",
) -> None:
    client = bigquery.Client(project=project_id)

    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        ds = bigquery.Dataset(dataset_ref)
        ds.location = location
        client.create_dataset(ds, exists_ok=True)
        print(f"Created dataset: {dataset_ref}")

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"Uploaded {len(df):,} rows → {table_ref}")


def save_local_csvs(train: pd.DataFrame, test: pd.DataFrame, inference: pd.DataFrame) -> None:
    out_dir = os.path.dirname(__file__)
    train.to_csv(os.path.join(out_dir, "train_data.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test_data.csv"), index=False)
    inference.to_csv(os.path.join(out_dir, "inference_data.csv"), index=False)
    print("\nLocal CSVs saved to data/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and upload used-car data")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--dataset", default="used_car_mlops", help="BigQuery dataset")
    parser.add_argument("--location", default="US", help="BQ dataset location")
    parser.add_argument("--local-only", action="store_true", help="Skip BigQuery upload")
    args = parser.parse_args()

    print("=== Generating synthetic used-car price data ===\n")
    train, test, inference = generate_datasets()
    save_local_csvs(train, test, inference)

    if not args.local_only:
        print("\n=== Uploading to BigQuery ===\n")
        upload_to_bigquery(train,     args.project, args.dataset, "car_prices_train",     args.location)
        upload_to_bigquery(test,      args.project, args.dataset, "car_prices_test",      args.location)
        upload_to_bigquery(inference, args.project, args.dataset, "car_prices_inference", args.location)

        from google.cloud import bigquery as bq
        client = bq.Client(project=args.project)
        preds_schema = [
            bq.SchemaField("row_id",        "STRING"),
            bq.SchemaField("predicted_price","FLOAT64"),
            bq.SchemaField("model_version",  "STRING"),
            bq.SchemaField("pipeline_run_id","STRING"),
            bq.SchemaField("prediction_ts",  "TIMESTAMP"),
        ]
        table_ref = f"{args.project}.{args.dataset}.car_predictions"
        table = bq.Table(table_ref, schema=preds_schema)
        client.create_table(table, exists_ok=True)
        print(f"Ensured predictions table: {table_ref}")

        preds_processed_schema = [
            bq.SchemaField("row_id",        "STRING"),
            bq.SchemaField("processed_ts",  "TIMESTAMP"),
        ]
        processed_ref = f"{args.project}.{args.dataset}.car_prices_inference_processed"
        processed_table = bq.Table(processed_ref, schema=preds_processed_schema)
        client.create_table(processed_table, exists_ok=True)
        print(f"Ensured processed-tracking table: {processed_ref}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
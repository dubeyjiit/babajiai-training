from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "scikit-learn==1.4.2",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "joblib==1.4.2",
        "google-cloud-aiplatform==1.60.0",
        "google-cloud-storage==2.17.0",
    ],
)
def data_preprocessing_op(
    project_id: str,
    region: str,
    experiment_name: str,
    experiment_run_name: str,
    staging_bucket: str,
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    preprocessed_train: Output[Dataset],
    preprocessed_test: Output[Dataset],
    preprocessor_artifact: Output[Model],
    preprocessing_metrics: Output[Metrics],
) -> None:
    """Fit a ColumnTransformer on train, transform train+test, log statistics."""
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    import google.cloud.aiplatform as aip

    aip.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
        experiment=experiment_name,
    )

    NUMERIC_FEATURES = ["year", "mileage", "engine_size", "horsepower",
                        "accident_count", "num_owners"]
    CATEGORICAL_FEATURES = ["fuel_type", "transmission", "brand", "condition"]
    TARGET = "price"

    print("=== Loading datasets ===")
    train_df = pd.read_csv(train_data.path + ".csv")
    test_df  = pd.read_csv(test_data.path  + ".csv")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # ── Feature engineering ────────────────────────────────────────────────────
    for df in [train_df, test_df]:
        df["car_age"] = 2024 - df["year"]
        df["mileage_per_year"] = df["mileage"] / (df["car_age"].replace(0, 1))
        df["hp_per_liter"] = df["horsepower"] / df["engine_size"]

    NUMERIC_FEATURES = NUMERIC_FEATURES + ["car_age", "mileage_per_year", "hp_per_liter"]

    # ── Build preprocessor ─────────────────────────────────────────────────────
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,       NUMERIC_FEATURES),
        ("cat", categorical_pipeline,   CATEGORICAL_FEATURES),
    ], remainder="drop")

    # ── Fit on train, transform both ───────────────────────────────────────────
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET]
    X_test  = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test  = test_df[TARGET]

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    feature_names = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    train_out = pd.DataFrame(X_train_t, columns=feature_names)
    train_out[TARGET] = y_train.values
    test_out  = pd.DataFrame(X_test_t,  columns=feature_names)
    test_out[TARGET]  = y_test.values

    train_out.to_csv(preprocessed_train.path + ".csv", index=False)
    test_out.to_csv(preprocessed_test.path   + ".csv", index=False)

    # ── Save fitted preprocessor ───────────────────────────────────────────────
    preprocessor_path = preprocessor_artifact.path + ".pkl"
    meta = {
        "numeric_features":     NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target":               TARGET,
        "feature_names":        feature_names,
    }
    joblib.dump({"preprocessor": preprocessor, "meta": meta}, preprocessor_path)
    print(f"Preprocessor saved → {preprocessor_path}")

    # ── Also copy preprocessor to a FIXED well-known GCS path ─────────────────
    # The KFP artifact path contains a run-specific hash; this fixed path lets
    # the inference pipeline and Cloud Function always find the latest preprocessor.
    fixed_gcs_path = f"{staging_bucket}/models/latest/preprocessor.pkl"
    try:
        from google.cloud import storage as gcs
        bucket_name = staging_bucket.replace("gs://", "").split("/")[0]
        blob_path   = "/".join(fixed_gcs_path.replace("gs://", "").split("/")[1:])
        gcs_client  = gcs.Client(project=project_id)
        gcs_client.bucket(bucket_name).blob(blob_path).upload_from_filename(preprocessor_path)
        print(f"Preprocessor also uploaded → {fixed_gcs_path}")
    except Exception as e:
        print(f"WARNING: Could not upload preprocessor to fixed path (non-fatal): {e}")

    # ── Log metrics to Vertex AI Experiments ──────────────────────────────────
    stats = {
        "train_rows":       int(len(train_df)),
        "test_rows":        int(len(test_df)),
        "num_features":     len(NUMERIC_FEATURES),
        "cat_features":     len(CATEGORICAL_FEATURES),
        "target_mean_train": float(y_train.mean()),
        "target_std_train":  float(y_train.std()),
        "target_min_train":  float(y_train.min()),
        "target_max_train":  float(y_train.max()),
        "missing_pct_train": float(train_df.isnull().sum().sum() / train_df.size * 100),
    }
    # Log to KFP metrics artifact
    for k, v in stats.items():
        preprocessing_metrics.log_metric(k, v)

    print("Preprocessing statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("=== Preprocessing complete ===")
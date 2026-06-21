"""
KFP component: Model Training

Trains an XGBoost model wrapped in a full sklearn Pipeline
(preprocessor + model) so the artifact is self-contained.
Every run logs to a Vertex AI Experiment and to ML Metadata.
"""

from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact


@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "scikit-learn==1.4.2",
        "xgboost==2.0.3",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "joblib==1.4.2",
        "google-cloud-aiplatform==1.60.0",
    ],
)
def model_training_op(
    project_id: str,
    region: str,
    staging_bucket: str,
    experiment_name: str,
    experiment_run_name: str,
    preprocessed_train: Input[Dataset],
    preprocessor_artifact: Input[Model],
    # Hyperparameters
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: int,
    # Outputs
    trained_model: Output[Model],
    training_metrics: Output[Metrics],
    feature_importance: Output[Artifact],
) -> None:
    """Train XGBoost, log to Vertex AI Experiments, save full sklearn Pipeline."""
    import json
    import time
    import uuid
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import google.cloud.aiplatform as aip

    aip.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
        experiment=experiment_name,
    )

    TARGET = "price"

    print("=== Loading preprocessed training data ===")
    train_df = pd.read_csv(preprocessed_train.path + ".csv")
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    print(f"Training on {len(X_train):,} rows × {X_train.shape[1]} features")

    # Load fitted preprocessor metadata for feature names
    preprocessor_data = joblib.load(preprocessor_artifact.path + ".pkl")
    meta = preprocessor_data["meta"]

    # ── Start Vertex AI Experiment Run ─────────────────────────────────────────
    # Append a short UUID so component retries never collide on the same context name.
    unique_run_name = f"{experiment_run_name}-{uuid.uuid4().hex[:6]}"
    print(f"Experiment run name: {unique_run_name}")
    with aip.start_run(unique_run_name):
        params = {
            "n_estimators":      n_estimators,
            "max_depth":         max_depth,
            "learning_rate":     learning_rate,
            "subsample":         subsample,
            "colsample_bytree":  colsample_bytree,
            "min_child_weight":  min_child_weight,
        }
        aip.log_params(params)

        print(f"=== Training XGBoost with params: {params} ===")
        t0 = time.time()

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            eval_metric="rmse",
        )

        # Fit with early-stopping eval set (last 10% of train as internal validation)
        split = int(0.9 * len(X_train))
        X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
        y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        elapsed = time.time() - t0
        print(f"Training completed in {elapsed:.1f}s")

        # ── Training metrics ───────────────────────────────────────────────────
        y_pred_train = model.predict(X_train)
        rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        mae_train  = float(mean_absolute_error(y_train, y_pred_train))
        r2_train   = float(r2_score(y_train, y_pred_train))

        metrics_dict = {
            "train_rmse":         rmse_train,
            "train_mae":          mae_train,
            "train_r2":           r2_train,
            "training_time_secs": round(elapsed, 2),
            "best_iteration":     int(model.best_iteration) if hasattr(model, "best_iteration") else n_estimators,
        }
        aip.log_metrics(metrics_dict)
        print("Training metrics:", metrics_dict)

        # Log to KFP Metrics artifact
        for k, v in metrics_dict.items():
            training_metrics.log_metric(k, v)

    # ── Save full pipeline (preprocessor + model) ─────────────────────────────
    # We bundle preprocessor + xgb so the artifact is self-contained for serving
    preprocessor = preprocessor_data["preprocessor"]
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        model),
    ])

    model_path = trained_model.path + ".pkl"
    joblib.dump(full_pipeline, model_path)
    print(f"Full sklearn Pipeline saved → {model_path}")

    # ── Feature importance ─────────────────────────────────────────────────────
    feature_names = meta["feature_names"]
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    with open(feature_importance.path + ".json", "w") as f:
        json.dump(importance_sorted, f, indent=2)

    print("\nTop-10 feature importances:")
    for feat, score in list(importance_sorted.items())[:10]:
        print(f"  {feat}: {score:.4f}")

    print("=== Training complete ===")
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple


@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "scikit-learn==1.4.2",
        "xgboost==2.0.3",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "joblib==1.4.2",
        "google-cloud-aiplatform==1.60.0",
        "matplotlib==3.9.0",
    ],
)
def model_evaluation_op(
    project_id: str,
    region: str,
    staging_bucket: str,
    experiment_name: str,
    experiment_run_name: str,
    rmse_threshold: float,
    r2_threshold: float,
    trained_model: Input[Model],
    preprocessed_test: Input[Dataset],
    feature_importance_artifact: Input[Model],
    evaluation_metrics: Output[Metrics],
) -> NamedTuple("EvalOutputs", [("deploy_decision", str), ("test_rmse", float), ("test_r2", float)]):
    """Evaluate model on test set, gate deployment, log to Vertex AI Metadata."""
    import uuid
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        mean_absolute_percentage_error,
    )
    import google.cloud.aiplatform as aip
    from collections import namedtuple

    aip.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
        experiment=experiment_name,
    )

    TARGET = "price"

    print("=== Loading model and test data ===")
    pipeline = joblib.load(trained_model.path + ".pkl")
    test_df  = pd.read_csv(preprocessed_test.path + ".csv")

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    print(f"Evaluating on {len(X_test):,} test rows")

    # ── Predictions ────────────────────────────────────────────────────────────
    # The pipeline includes preprocessor, so we need raw features
    # But preprocessed_test already has transformed features → use model directly
    model = pipeline.named_steps["model"]
    y_pred = model.predict(X_test.values)

    # ── Regression metrics ─────────────────────────────────────────────────────
    rmse  = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae   = float(mean_absolute_error(y_test, y_pred))
    r2    = float(r2_score(y_test, y_pred))
    mape  = float(mean_absolute_percentage_error(y_test, y_pred)) * 100
    # Residuals
    residuals = y_pred - y_test.values
    mean_resid = float(residuals.mean())
    std_resid  = float(residuals.std())

    # Price bucket accuracy (within ±$2000)
    within_2k = float(np.mean(np.abs(residuals) <= 2_000) * 100)

    metrics = {
        "test_rmse":              rmse,
        "test_mae":               mae,
        "test_r2":                r2,
        "test_mape_pct":          round(mape, 2),
        "test_mean_residual":     round(mean_resid, 2),
        "test_std_residual":      round(std_resid, 2),
        "test_within_2k_pct":     round(within_2k, 2),
        "rmse_threshold":         rmse_threshold,
        "r2_threshold":           r2_threshold,
    }

    print("\n=== Evaluation metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        evaluation_metrics.log_metric(k, v)

    # ── Deployment gate ────────────────────────────────────────────────────────
    passes_rmse = rmse <= rmse_threshold
    passes_r2   = r2   >= r2_threshold
    deploy_decision = "deploy" if (passes_rmse and passes_r2) else "reject"

    print("\n=== Deployment gate ===")
    print(f"  RMSE {rmse:.0f} <= threshold {rmse_threshold:.0f}: {passes_rmse}")
    print(f"  R²   {r2:.3f} >= threshold {r2_threshold:.3f}: {passes_r2}")
    print(f"  Decision: {deploy_decision.upper()}")

    evaluation_metrics.log_metric("deploy_decision", deploy_decision)

    # ── Log to Vertex AI Experiment Run ───────────────────────────────────────
    # Unique suffix prevents AlreadyExists 409 on component retries.
    unique_run_name = f"{experiment_run_name}-eval-{uuid.uuid4().hex[:6]}"
    print(f"Experiment run name: {unique_run_name}")
    with aip.start_run(unique_run_name):
        aip.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        aip.log_params({"deploy_decision": deploy_decision})

    EvalOutputs = namedtuple("EvalOutputs", ["deploy_decision", "test_rmse", "test_r2"])
    return EvalOutputs(
        deploy_decision=deploy_decision,
        test_rmse=round(rmse, 2),
        test_r2=round(r2, 4),
    )

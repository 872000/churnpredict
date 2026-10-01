"""Real tests for the ChurnPredict pipeline."""

import pandas as pd
import pytest

from churnpredict import (
    FEATURE_COLUMNS,
    TARGET,
    generate_dataset,
    load_model,
    save_model,
    train_models,
)


def test_generate_dataset_shape_and_labels():
    df = generate_dataset(n_samples=500, seed=123)
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET]
    assert len(df) == 500
    assert set(df[TARGET].unique()) <= {0, 1}
    # churn should be a minority but not degenerate
    rate = df[TARGET].mean()
    assert 0.05 < rate < 0.60


def test_generate_dataset_is_seeded():
    a = generate_dataset(n_samples=200, seed=7)
    b = generate_dataset(n_samples=200, seed=7)
    c = generate_dataset(n_samples=200, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_train_models_runs_and_metrics_are_valid():
    df = generate_dataset(n_samples=400, seed=42)
    results = train_models(df, seed=42, rf_n_estimators=20)

    assert set(results["models"]) == {"logistic_regression", "random_forest"}
    assert results["best"] in results["models"]

    for name, m in results["metrics"].items():
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= m[key] <= 1.0, f"{name}.{key} out of range"
        # models should beat a coin flip on this synthetic data
        assert m["roc_auc"] > 0.70, f"{name} roc_auc too low"
        cm = m["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2
        assert sum(sum(row) for row in cm) == len(results["y_test"])


def test_save_and_load_model(tmp_path):
    df = generate_dataset(n_samples=300, seed=42)
    results = train_models(df, seed=42, rf_n_estimators=10)
    path = tmp_path / "model.joblib"
    save_model({"pipeline": results["models"]["logistic_regression"]}, path)
    bundle = load_model(path)
    assert hasattr(bundle["pipeline"], "predict")


def test_score_single_customer():
    df = generate_dataset(n_samples=300, seed=42)
    results = train_models(df, seed=42, rf_n_estimators=10)
    pipeline = results["models"]["logistic_regression"]

    customer = pd.DataFrame([{
        "tenure_months": 4,
        "monthly_charges": 89.99,
        "support_calls": 5,
        "avg_monthly_gb": 210.5,
        "num_late_payments": 2,
        "senior_citizen": 1,
        "has_dependents": 0,
        "has_partner": 0,
        "paperless_billing": 1,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }])

    proba = pipeline.predict_proba(customer)[0, 1]
    pred = pipeline.predict(customer)[0]
    assert 0.0 <= proba <= 1.0
    assert pred in (0, 1)
    # a risky-profile customer should score high churn probability
    assert proba > 0.5


def test_cli_score_accepts_long_json_string(tmp_path, capsys):
    """`score --json` must accept an inline JSON string, not just a file."""
    import json

    from churnpredict.cli import main

    df = generate_dataset(n_samples=200, seed=42)
    results = train_models(df, seed=42, rf_n_estimators=10)
    model_path = tmp_path / "model.joblib"
    save_model(
        {
            "pipeline": results["models"]["logistic_regression"],
            "model_name": "logistic_regression",
            "metrics": results["metrics"]["logistic_regression"],
            "feature_columns": FEATURE_COLUMNS,
            "target": TARGET,
        },
        model_path,
    )

    customer = {
        "tenure_months": 4,
        "monthly_charges": 89.99,
        "support_calls": 5,
        "avg_monthly_gb": 210.5,
        "num_late_payments": 2,
        "senior_citizen": 1,
        "has_dependents": 0,
        "has_partner": 0,
        "paperless_billing": 1,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
    }
    rc = main(["score", "--model-path", str(model_path), "--json", json.dumps(customer)])
    assert rc == 0
    assert "churn_probability" in capsys.readouterr().out

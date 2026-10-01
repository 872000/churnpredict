"""Model training, evaluation and plotting for the churn pipeline."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe: never opens a GUI window

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .data import TARGET
from .features import make_pipeline, transformed_feature_names

MODEL_SPECS = {
    "logistic_regression": lambda seed: LogisticRegression(
        max_iter=1_000, class_weight="balanced", random_state=seed
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ),
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    """Compute the standard classification metrics (all in [0, 1])."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=3),
    }


def train_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    rf_n_estimators: int | None = None,
) -> dict[str, Any]:
    """Stratified train/test split, train both models, evaluate, pick a winner.

    Returns a dict with fitted pipelines, per-model metrics, the test split
    and the name of the best model (highest F1 on the held-out test set).
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    specs = dict(MODEL_SPECS)
    if rf_n_estimators is not None:
        specs["random_forest"] = lambda s: RandomForestClassifier(
            n_estimators=rf_n_estimators,
            class_weight="balanced",
            random_state=s,
            n_jobs=-1,
        )

    models: dict[str, Any] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for name, make_clf in specs.items():
        pipeline = make_pipeline(make_clf(seed))
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        models[name] = pipeline
        metrics[name] = evaluate(y_test, y_pred, y_proba)

    best = max(metrics, key=lambda name: metrics[name]["f1"])
    return {
        "models": models,
        "metrics": metrics,
        "best": best,
        "X_test": X_test,
        "y_test": y_test,
    }


def print_results(results: dict[str, Any]) -> None:
    """Print a clean model comparison to stdout."""
    metrics = results["metrics"]
    header = f"{'model':<20} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'roc-auc':>7}"
    print("\nModel comparison (held-out test set)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(
            f"{name:<20} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['roc_auc']:>7.3f}"
        )
    print("-" * len(header))
    print(f"Best model by F1: {results['best']}\n")
    for name, m in metrics.items():
        print(f"--- classification report: {name} ---")
        print(m["classification_report"])


def plot_roc_curve(
    models: dict[str, Any], X_test: pd.DataFrame, y_test: np.ndarray, path: str | Path
) -> Path:
    """ROC curve for every model, saved as a PNG."""
    path = Path(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pipeline in models.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title("ROC curves — customer churn models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    pipeline, path: str | Path, top_n: int = 12
) -> Path:
    """Random-forest feature-importance bar chart, saved as a PNG."""
    path = Path(path)
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise ValueError("pipeline classifier has no feature_importances_")
    names = transformed_feature_names(pipeline)
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    labels = [names[i].replace("num__", "").replace("cat__", "") for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], importances[order][::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title("Random forest — top churn predictors")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_model(bundle: dict[str, Any], path: str | Path) -> Path:
    """Persist a trained pipeline + metadata with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a bundle saved with :func:`save_model`."""
    return joblib.load(path)

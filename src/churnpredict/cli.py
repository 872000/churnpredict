"""Command-line interface: train, score and demo subcommands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data import FEATURE_COLUMNS, TARGET, describe_churn_mechanism, generate_dataset
from .modeling import (
    load_model,
    plot_feature_importance,
    plot_roc_curve,
    print_results,
    save_model,
    train_models,
)

DEFAULT_MODEL_PATH = "models/churn_model.joblib"
DEFAULT_DOCS_DIR = "docs"


def _resolve_json_input(raw: str):
    """Accept a path to a JSON file, or a JSON string (object or list)."""
    try:
        candidate = Path(raw)
        is_path = candidate.exists()
    except OSError:  # e.g. the JSON string is too long to be a filename
        is_path = False
    if is_path:
        raw = candidate.read_text()
    payload = json.loads(raw)
    return payload if isinstance(payload, list) else [payload]


def cmd_train(args: argparse.Namespace) -> int:
    df = generate_dataset(n_samples=args.n_samples, seed=args.seed)
    print(f"Generated {len(df):,} synthetic customers (seed={args.seed}).")
    print(f"Observed churn rate: {df[TARGET].mean():.1%}")
    print(f"Churn mechanism: {describe_churn_mechanism()}\n")

    results = train_models(df, test_size=args.test_size, seed=args.seed)
    print_results(results)

    docs = Path(args.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    roc_path = plot_roc_curve(
        results["models"], results["X_test"], results["y_test"], docs / "roc_curve.png"
    )
    fi_path = plot_feature_importance(
        results["models"]["random_forest"], docs / "feature_importance.png"
    )
    print(f"Saved ROC curve:            {roc_path}")
    print(f"Saved feature importance:   {fi_path}")

    bundle = {
        "pipeline": results["models"][results["best"]],
        "model_name": results["best"],
        "metrics": results["metrics"][results["best"]],
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "train_seed": args.seed,
        "n_samples": args.n_samples,
    }
    model_path = save_model(bundle, args.model_path)
    print(f"Saved best model ({results['best']}): {model_path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    bundle = load_model(args.model_path)
    pipeline = bundle["pipeline"]

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        records = _resolve_json_input(args.json)
        df = pd.DataFrame(records)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: missing feature columns: {missing}", file=sys.stderr)
        return 2

    df = df[FEATURE_COLUMNS]
    proba = pipeline.predict_proba(df)[:, 1]
    pred = pipeline.predict(df)

    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["predicted_churn"] = pred
    print(f"Scored with model: {bundle.get('model_name', 'unknown')}")
    print(out[["churn_probability", "predicted_churn"] + FEATURE_COLUMNS[:3]].to_string(index=False))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    print("=== ChurnPredict demo: synthetic data → train → evaluate → save ===\n")
    train_args = argparse.Namespace(
        n_samples=1_500,
        seed=42,
        test_size=0.2,
        model_path=args.model_path,
        docs_dir=args.docs_dir,
    )
    return cmd_train(train_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churnpredict",
        description="Customer churn ML pipeline: train, evaluate and explain with scikit-learn.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train both models, evaluate, save the best.")
    p_train.add_argument("--n-samples", type=int, default=5_000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p_train.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    p_train.set_defaults(func=cmd_train)

    p_score = sub.add_parser("score", help="Score new customers from CSV or JSON.")
    p_score.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    group = p_score.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Path to CSV with one customer per row.")
    group.add_argument("--json", help="JSON file path or JSON string (object or list).")
    p_score.set_defaults(func=cmd_score)

    p_demo = sub.add_parser("demo", help="Quick end-to-end run on small synthetic data.")
    p_demo.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p_demo.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""ChurnPredict — customer churn ML pipeline built on scikit-learn.

Generate a realistic synthetic telecom dataset, train and compare
logistic regression and random forest classifiers, evaluate them with
precision / recall / F1 / ROC-AUC, and score new customers from the CLI.
"""

__version__ = "1.0.0"

from .data import TARGET, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, FEATURE_COLUMNS, generate_dataset
from .features import build_preprocessor
from .modeling import train_models, evaluate, save_model, load_model

__all__ = [
    "__version__",
    "TARGET",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "FEATURE_COLUMNS",
    "generate_dataset",
    "build_preprocessor",
    "train_models",
    "evaluate",
    "save_model",
    "load_model",
]

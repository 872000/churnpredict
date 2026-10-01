"""Feature preprocessing for the churn pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing step shared by every model.

    Numeric features are standardized; categoricals are one-hot encoded
    (unknown categories are ignored so scoring tolerates new values).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_pipeline(classifier) -> Pipeline:
    """Wrap a classifier in the standard preprocessing pipeline."""
    return Pipeline([("preprocess", build_preprocessor()), ("clf", classifier)])


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Feature names after the fitted preprocessor (one-hot expanded)."""
    preprocessor = pipeline.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())

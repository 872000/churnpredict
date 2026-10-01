"""Seeded synthetic telecom-customer dataset generator.

Every call with the same ``seed`` produces the identical dataset, so the
pipeline is fully reproducible. The churn mechanism is a logistic model
with plausible drivers:

* month-to-month contracts and fiber-optic plans churn more,
* higher monthly charges, more support calls and late payments raise churn,
* longer tenure, partner/dependents and long-term contracts lower churn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "churn"

CATEGORICAL_FEATURES = ["contract_type", "internet_service", "payment_method"]

NUMERICAL_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "avg_monthly_gb",
    "num_late_payments",
    "senior_citizen",
    "has_dependents",
    "has_partner",
    "paperless_billing",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_dataset(n_samples: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic telecom customer churn dataset.

    Parameters
    ----------
    n_samples: number of customers to simulate.
    seed: random seed; identical seeds give identical datasets.

    Returns
    -------
    DataFrame with ``FEATURE_COLUMNS`` plus the binary ``churn`` target.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.exponential(18, n_samples).astype(int), 0, 72)
    fiber = rng.random(n_samples) < 0.45

    monthly_charges = np.clip(
        rng.normal(62, 18, n_samples) + np.where(fiber, 14, 0), 18.0, 120.0
    ).round(2)

    contract = rng.choice(
        CONTRACT_TYPES, n_samples, p=[0.45, 0.30, 0.25]
    )
    internet = np.where(
        fiber,
        "Fiber optic",
        rng.choice(["DSL", "No"], n_samples, p=[0.64, 0.36]),
    )
    payment = rng.choice(
        PAYMENT_METHODS, n_samples, p=[0.34, 0.20, 0.23, 0.23]
    )

    lam_calls = 0.7 + 0.6 * fiber + 0.02 * np.clip(monthly_charges - 65, 0, None)
    support_calls = np.clip(rng.poisson(lam_calls), 0, 12)

    avg_monthly_gb = np.clip(rng.exponential(120, n_samples).round(1), 1.0, 1500.0)
    num_late_payments = np.clip(rng.poisson(0.5, n_samples), 0, 6)

    senior_citizen = rng.binomial(1, 0.16, n_samples)
    has_dependents = rng.binomial(1, 0.30, n_samples)
    has_partner = rng.binomial(1, 0.45, n_samples)
    paperless_billing = rng.binomial(1, 0.60, n_samples)

    # --- churn mechanism: logistic score over the features -----------------
    logit = (
        -1.60
        + 1.35 * (contract == "Month-to-month")
        - 0.55 * (contract == "Two year")
        + 0.55 * (internet == "Fiber optic")
        - 0.35 * (internet == "No")
        + 0.028 * (monthly_charges - 65)
        + 0.42 * support_calls
        - 0.035 * tenure
        + 0.35 * senior_citizen
        + 0.55 * num_late_payments
        + 0.45 * (payment == "Electronic check")
        - 0.35 * has_partner
        - 0.25 * has_dependents
    )
    churn = rng.binomial(1, _sigmoid(logit))

    df = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "avg_monthly_gb": avg_monthly_gb,
            "num_late_payments": num_late_payments,
            "senior_citizen": senior_citizen,
            "has_dependents": has_dependents,
            "has_partner": has_partner,
            "paperless_billing": paperless_billing,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "churn": churn,
        }
    )
    return df[FEATURE_COLUMNS + [TARGET]]


def describe_churn_mechanism() -> str:
    """Human-readable summary of the synthetic churn drivers."""
    return (
        "Churn probability rises with month-to-month contracts, fiber-optic plans, "
        "higher monthly charges, more support calls, late payments, senior status "
        "and electronic-check billing; it falls with longer tenure, two-year "
        "contracts, no internet service, partners and dependents."
    )

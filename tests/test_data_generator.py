"""
Tests for src/data_generator.py — SyntheticCustomerGenerator.

Each test focuses on a single, independently verifiable property of the
generated DataFrame so that failures give a precise signal about what broke.
"""

import pytest

from src.data_generator import SyntheticCustomerGenerator

VALID_SEGMENTS = {"high_value", "standard", "churn_risk"}
EXPECTED_PROPORTIONS = {"high_value": 0.20, "standard": 0.50, "churn_risk": 0.30}
PROPORTION_TOLERANCE = 0.05   # ± 5 percentage points


# ---------------------------------------------------------------------------
# Row count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 50, 500, 1_000])
def test_generate_returns_correct_number_of_rows(n):
    """generate() must return exactly n_customers rows, for any valid n."""
    gen = SyntheticCustomerGenerator(n_customers=n, seed=0)
    df = gen.generate()
    assert len(df) == n


# ---------------------------------------------------------------------------
# Segment validity
# ---------------------------------------------------------------------------

def test_segment_values_are_valid():
    """Every value in the segment column must be one of the three known labels."""
    gen = SyntheticCustomerGenerator(n_customers=1_000, seed=42)
    df = gen.generate()
    unexpected = set(df["segment"].unique()) - VALID_SEGMENTS
    assert unexpected == set(), f"Unexpected segment labels: {unexpected}"


# ---------------------------------------------------------------------------
# Monthly spending
# ---------------------------------------------------------------------------

def test_monthly_spending_is_always_positive():
    """Gamma-distributed spend must be strictly positive for all customers."""
    gen = SyntheticCustomerGenerator(n_customers=1_000, seed=42)
    df = gen.generate()
    assert (df["monthly_spending"] > 0).all(), (
        f"Found {(df['monthly_spending'] <= 0).sum()} non-positive spending values"
    )


# ---------------------------------------------------------------------------
# Months to churn
# ---------------------------------------------------------------------------

def test_months_to_churn_is_non_negative():
    """months_to_churn must be >= 0 for all customers.

    The raw Weibull draw is always strictly positive, but the generator rounds
    to one decimal place (`.round(1)`).  This means very small draws (< 0.05)
    become exactly 0.0 — representing a customer who churns before their first
    full month.  Negative values are impossible given the Weibull distribution.

    Note: values < 1 are valid and common for the churn_risk segment; the
    test checks the mathematical floor, not a business minimum tenure.
    """
    gen = SyntheticCustomerGenerator(n_customers=1_000, seed=42)
    df = gen.generate()
    assert (df["months_to_churn"] >= 0).all(), (
        f"Found {(df['months_to_churn'] < 0).sum()} negative churn months"
    )


# ---------------------------------------------------------------------------
# Segment proportions
# ---------------------------------------------------------------------------

def test_segment_proportions_are_approximately_correct():
    """With 10,000 customers the empirical mix must be within 5 pp of the target.

    Uses a large sample so the law-of-large-numbers guarantee is tight enough
    that ± 5 pp is a safe tolerance (observed std ≈ 0.5 pp at n=10 000).
    """
    gen = SyntheticCustomerGenerator(n_customers=10_000, seed=42)
    df = gen.generate()
    proportions = df["segment"].value_counts(normalize=True)

    for segment, expected in EXPECTED_PROPORTIONS.items():
        actual = proportions[segment]
        assert abs(actual - expected) < PROPORTION_TOLERANCE, (
            f"Segment '{segment}': expected ~{expected:.0%}, "
            f"got {actual:.2%} (tolerance ± {PROPORTION_TOLERANCE:.0%})"
        )

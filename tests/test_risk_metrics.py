"""
Tests for src/risk_metrics.py — RiskMetrics.

Uses the shared `risk_metrics` fixture (session-scoped) from conftest.py so
the expensive simulation is run once and reused across all tests in this file.
"""

import pytest

from src.risk_metrics import RiskMetrics

# ---------------------------------------------------------------------------
# Expected interface contracts
# ---------------------------------------------------------------------------

PORTFOLIO_SUMMARY_KEYS = {
    "median_clv",
    "mean_clv",
    "clv_5th_pct",
    "clv_95th_pct",
    "pct_negative_p5",
}

EXPECTED_SEGMENTS = {"high_value", "standard", "churn_risk"}


# ---------------------------------------------------------------------------
# optimal_cac
# ---------------------------------------------------------------------------

def test_optimal_cac_is_less_than_median_clv(risk_metrics):
    """optimal_cac() partitions median CLV over payback_months, so the result
    must always be strictly less than the full median CLV.

    This is a sanity check on the business logic: you should never spend more
    to acquire a customer than you expect to earn from them.
    """
    median_clv = risk_metrics.results["clv_p50"].median()
    cac = risk_metrics.optimal_cac(payback_months=36)

    assert cac < median_clv, (
        f"optimal_cac ({cac:.2f}) must be < median CLV ({median_clv:.2f})"
    )


def test_optimal_cac_is_positive(risk_metrics):
    """CAC must be positive — a negative acquisition budget makes no sense."""
    cac = risk_metrics.optimal_cac(payback_months=36)
    assert cac > 0, f"Expected positive CAC, got {cac:.4f}"


# ---------------------------------------------------------------------------
# portfolio_summary
# ---------------------------------------------------------------------------

def test_portfolio_summary_returns_all_required_keys(risk_metrics):
    """portfolio_summary() must return every documented key."""
    summary = risk_metrics.portfolio_summary()
    missing = PORTFOLIO_SUMMARY_KEYS - set(summary.keys())
    assert missing == set(), f"portfolio_summary() is missing keys: {missing}"


def test_portfolio_summary_values_are_numeric(risk_metrics):
    """All values in the summary dict must be real numbers (not NaN/None)."""
    summary = risk_metrics.portfolio_summary()
    for key, value in summary.items():
        assert isinstance(value, (int, float)), (
            f"Key '{key}' has non-numeric value: {value!r}"
        )
        assert value == value, f"Key '{key}' is NaN"   # NaN != NaN


# ---------------------------------------------------------------------------
# segment_breakdown
# ---------------------------------------------------------------------------

def test_segment_breakdown_has_one_row_per_segment(risk_metrics):
    """segment_breakdown() must return exactly one row for each of the three
    known segments (assuming all three are present in the 200-customer sample).
    """
    breakdown = risk_metrics.segment_breakdown()
    returned_segments = set(breakdown.index)
    assert returned_segments == EXPECTED_SEGMENTS, (
        f"Expected segments {EXPECTED_SEGMENTS}, got {returned_segments}"
    )


def test_segment_breakdown_columns_are_present(risk_metrics):
    """segment_breakdown() must expose the documented column set."""
    expected_cols = {"n_customers", "median_clv", "std_clv", "p5_clv", "p95_clv"}
    breakdown = risk_metrics.segment_breakdown()
    missing = expected_cols - set(breakdown.columns)
    assert missing == set(), f"segment_breakdown() is missing columns: {missing}"

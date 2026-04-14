"""
Tests for src/simulation.py — CLVSimulation and MCConfig.

Strategy: use tiny customer sets and a small n_simulations so the test suite
runs in a few seconds.  The numeric properties being checked (sign, ordering,
monotonicity) are robust to sample size — we don't need 100,000 paths to
verify that a zero-spend customer has CLV = 0.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation import CLVSimulation, MCConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_customer(
    customer_id: int = 1,
    segment: str = "standard",
    monthly_spending: float = 200.0,
    months_to_churn: float = 24.0,
) -> pd.Series:
    """Return a minimal customer Series compatible with CLVSimulation."""
    return pd.Series(
        {
            "customer_id": customer_id,
            "segment": segment,
            "monthly_spending": monthly_spending,
            "months_to_churn": months_to_churn,
        }
    )


def _make_customers_df(**kwargs) -> pd.DataFrame:
    """Wrap _make_customer() in a one-row DataFrame."""
    return pd.DataFrame([_make_customer(**kwargs)])


def _fast_config(**overrides) -> MCConfig:
    """MCConfig with 500 paths — fast enough for unit tests."""
    defaults = dict(n_simulations=500, planning_horizon_months=60,
                    discount_rate_annual=0.12, seed=42)
    defaults.update(overrides)
    return MCConfig(**defaults)


# ---------------------------------------------------------------------------
# simulate_customer_path — output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_sims", [1, 10, 500])
def test_simulate_customer_path_returns_correct_length(n_sims):
    """simulate_customer_path must return exactly n_sims values."""
    config = _fast_config(n_simulations=n_sims)
    sim = CLVSimulation(customers=_make_customers_df(), config=config)
    paths = sim.simulate_customer_path(_make_customer(), n_sims=n_sims)
    assert paths.shape == (n_sims,), (
        f"Expected shape ({n_sims},), got {paths.shape}"
    )


# ---------------------------------------------------------------------------
# simulate_customer_path — non-negativity
# ---------------------------------------------------------------------------

def test_clv_paths_are_non_negative():
    """All simulated CLV values must be >= 0.

    Spending draws are lognormal (always positive); the only way to get 0 is
    if a customer's churn draw is clipped to 0 (immediate churn).  Either way,
    negative CLV is impossible with this model.
    """
    config = _fast_config(n_simulations=1_000)
    sim = CLVSimulation(customers=_make_customers_df(), config=config)
    paths = sim.simulate_customer_path(_make_customer(), n_sims=1_000)
    assert (paths >= 0).all(), (
        f"Found {(paths < 0).sum()} negative CLV paths out of 1,000"
    )


# ---------------------------------------------------------------------------
# run_full_simulation — column contract
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "customer_id", "segment",
    "clv_p5", "clv_p25", "clv_p50", "clv_p75", "clv_p95",
    "clv_mean", "clv_std",
}


def test_run_full_simulation_returns_correct_columns():
    """The results DataFrame must contain exactly the documented columns."""
    customers = pd.DataFrame([
        _make_customer(customer_id=1),
        _make_customer(customer_id=2, segment="high_value", months_to_churn=48.0),
    ])
    config = _fast_config()
    sim = CLVSimulation(customers=customers, config=config)
    results = sim.run_full_simulation()
    assert set(results.columns) == EXPECTED_COLUMNS, (
        f"Missing: {EXPECTED_COLUMNS - set(results.columns)}  "
        f"Extra: {set(results.columns) - EXPECTED_COLUMNS}"
    )


# ---------------------------------------------------------------------------
# Churn horizon effect: short churn → lower median CLV
# ---------------------------------------------------------------------------

def test_short_churn_month_yields_lower_median_clv_than_long_churn():
    """A customer who churns after 3 months should have a lower median CLV
    than an otherwise identical customer who churns after 60 months.

    Both customers share the same spending level so any CLV difference is
    purely driven by time-in-portfolio.
    """
    SPEND = 300.0

    short_df = _make_customers_df(monthly_spending=SPEND, months_to_churn=3.0)
    long_df  = _make_customers_df(monthly_spending=SPEND, months_to_churn=60.0)

    config = _fast_config(n_simulations=2_000)

    sim_short = CLVSimulation(customers=short_df, config=config)
    paths_short = sim_short.simulate_customer_path(short_df.iloc[0], n_sims=2_000)

    sim_long = CLVSimulation(customers=long_df, config=config)
    paths_long = sim_long.simulate_customer_path(long_df.iloc[0], n_sims=2_000)

    assert np.median(paths_short) < np.median(paths_long), (
        f"Short-churn median CLV ({np.median(paths_short):.2f}) should be "
        f"< long-churn median CLV ({np.median(paths_long):.2f})"
    )


# ---------------------------------------------------------------------------
# Discount rate effect: higher rate → lower CLV
# ---------------------------------------------------------------------------

def test_higher_discount_rate_decreases_median_clv(customers_200):
    """Increasing the annual discount rate must reduce median portfolio CLV.

    The first 20 customers are used to keep runtime short.  The effect is
    deterministic in sign (NPV discounts future cash flows more aggressively
    at higher rates) so even a small sample is sufficient.
    """
    subset = customers_200.head(20).copy()

    low_rate_config  = _fast_config(discount_rate_annual=0.05, seed=42)
    high_rate_config = _fast_config(discount_rate_annual=0.30, seed=42)

    results_low  = CLVSimulation(customers=subset, config=low_rate_config).run_full_simulation()
    results_high = CLVSimulation(customers=subset, config=high_rate_config).run_full_simulation()

    median_low  = results_low["clv_p50"].median()
    median_high = results_high["clv_p50"].median()

    assert median_low > median_high, (
        f"Low-rate median CLV ({median_low:.2f}) should be "
        f"> high-rate median CLV ({median_high:.2f})"
    )

"""
Shared pytest fixtures for the CLV Monte Carlo test suite.

Fixtures defined here are automatically available to every test file in the
tests/ directory — no import needed.

Scope notes:
- scope="session": created once per test run and reused.  Used for objects
  that are expensive to build (e.g. a 200-customer simulation result).
- scope="function" (default): created fresh for every test.  Used when a test
  might mutate the object, so isolation matters.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import SyntheticCustomerGenerator
from src.simulation import CLVSimulation, MCConfig
from src.risk_metrics import RiskMetrics


# ---------------------------------------------------------------------------
# Customer data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def customers_200():
    """200-row customer DataFrame, fixed seed for reproducibility."""
    gen = SyntheticCustomerGenerator(n_customers=200, seed=42)
    return gen.generate()


# ---------------------------------------------------------------------------
# Simulation config — kept small so tests run in seconds, not minutes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fast_config():
    """MCConfig with 200 Monte Carlo paths per customer."""
    return MCConfig(
        n_simulations=200,
        planning_horizon_months=60,
        discount_rate_annual=0.12,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Simulation results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def simulation_results(customers_200, fast_config):
    """Pre-computed simulation results for 200 customers."""
    sim = CLVSimulation(customers=customers_200, config=fast_config)
    return sim.run_full_simulation()


# ---------------------------------------------------------------------------
# RiskMetrics instance
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def risk_metrics(simulation_results):
    """RiskMetrics built from the shared simulation results."""
    return RiskMetrics(simulation_results)

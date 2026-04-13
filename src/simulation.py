"""
Monte Carlo Customer Lifetime Value (CLV) simulation engine.

--------------------------------------------------------------------------------
What is NPV discounting?
--------------------------------------------------------------------------------
A rand received today is worth more than a rand received next year — because
today's rand can be invested and earn a return. Net Present Value (NPV)
discounting formalises this by converting every future cash flow back to its
equivalent value in today's money.

  Formula:
      NPV = Σ  [Cash_t / (1 + r)^t]
             t=1..T

  Where:
      Cash_t  = rand amount received at month t
      r       = monthly discount rate (e.g. 0.12 / 12 = 0.01 for 12 % p.a.)
      t       = number of months in the future

  Concrete example (r = 1 % / month, planning horizon = 12 months):
      A customer spends R200 per month. Undiscounted total = R2,400.

      Month  1  →  R200 / (1.01)^1  = R198.02
      Month  6  →  R200 / (1.01)^6  = R188.17
      Month 12  →  R200 / (1.01)^12 = R177.48
                                       --------
                   NPV total         ≈ R2,256 (6 % less than the face value)

  The further into the future a cash flow sits, the more it is discounted.
  This means a customer who spends R200/month for 12 months is *more* valuable
  than one who spends R200/month only in months 7–18 — even though the
  nominal totals are identical — because the early cash flows compound sooner.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCConfig:
    """Hyper-parameters for the Monte Carlo CLV simulation.

    Parameters
    ----------
    n_simulations:
        Number of Monte Carlo paths drawn per customer.  100,000 gives tight
        percentile estimates; drop to 1,000 for fast iteration during
        development, or to 1 for a single-path point estimate (see
        ``CLVSimulation.run_full_simulation`` docstring for the speed
        trade-off discussion).
    planning_horizon_months:
        How many months into the future to project cash flows (default: 60
        months = 5 years).
    discount_rate_annual:
        Annual discount rate as a decimal (0.12 = 12 %).  Converted to a
        monthly rate via the ``monthly_discount_rate`` property.
    seed:
        Random seed for reproducibility.
    """

    n_simulations: int = 100_000
    planning_horizon_months: int = 60
    discount_rate_annual: float = 0.12
    seed: int = 42

    @property
    def monthly_discount_rate(self) -> float:
        """Equivalent monthly rate derived from the annual rate.

        Uses the compound-interest conversion so that 12 monthly periods
        correctly reproduce the annual rate:

            r_monthly = (1 + r_annual)^(1/12) - 1

        Example: 12 % p.a.  →  (1.12)^(1/12) - 1  ≈  0.9489 % / month
        (NOT the naive 12/12 = 1 % which understates compounding).
        """
        return (1.0 + self.discount_rate_annual) ** (1.0 / 12.0) - 1.0


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

class CLVSimulation:
    """Vectorised Monte Carlo CLV simulator.

    For each customer the simulation draws ``n_simulations`` independent
    lifetime paths and accumulates discounted monthly spending up to the
    simulated churn event, producing a full CLV distribution per customer.

    Parameters
    ----------
    customers:
        DataFrame produced by ``SyntheticCustomerGenerator.generate()``.
        Required columns: ``customer_id``, ``segment``, ``monthly_spending``,
        ``months_to_churn``.
    config:
        ``MCConfig`` instance controlling simulation hyper-parameters.
    """

    def __init__(self, customers: pd.DataFrame, config: MCConfig) -> None:
        self.customers = customers.reset_index(drop=True)
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Core path simulator
    # ------------------------------------------------------------------

    def simulate_customer_path(
        self, customer: pd.Series, n_sims: int
    ) -> np.ndarray:
        """Simulate ``n_sims`` independent CLV paths for a single customer.

        Algorithm
        ---------
        1. **Churn month sampling** — the customer's base ``months_to_churn``
           is perturbed with Gaussian noise (std = 10 % of base) to model
           uncertainty in when they will actually leave::

               churn_t ~ Normal(months_to_churn, 0.1 * months_to_churn)

           Negative draws are clipped to 0 (immediate churn).

        2. **Monthly spend sampling** — for each month in the planning
           horizon where the customer is still active (month ≤ churn_t),
           a spending draw is taken from a Lognormal distribution centred on
           the customer's historical ``monthly_spending`` with a 10 % noise
           coefficient of variation::

               spend_t ~ LogNormal(μ, σ=0.1)
               where  μ = ln(monthly_spending) - σ²/2

           The μ parameterisation ensures E[spend_t] = monthly_spending.

        3. **NPV discounting** — each monthly cash flow is multiplied by the
           corresponding discount factor before summing::

               CLV = Σ active_t · spend_t · (1 + r)^{-t}

        Parameters
        ----------
        customer:
            A single row from the customers DataFrame (pd.Series).
        n_sims:
            Number of Monte Carlo paths to draw.

        Returns
        -------
        np.ndarray, shape (n_sims,)
            Present-value CLV for each simulated path.
        """
        H = self.config.planning_horizon_months
        r = self.config.monthly_discount_rate

        base_churn = float(customer["months_to_churn"])
        base_spend = float(customer["monthly_spending"])

        # --- 1. Churn month per path -------------------------------------------
        churn_noise_std = 0.10 * base_churn
        churn_months = self.rng.normal(base_churn, churn_noise_std, size=n_sims)
        churn_months = np.maximum(churn_months, 0.0)          # shape (n_sims,)

        # --- 2. Monthly spend draws (lognormal, ±10 % CV) ----------------------
        sigma = 0.10
        mu = np.log(base_spend) - 0.5 * sigma ** 2
        # shape (n_sims, H) — one draw per path per month
        spend = self.rng.lognormal(mean=mu, sigma=sigma, size=(n_sims, H))

        # --- 3. Active mask: customer alive in month t iff t <= churn_month ----
        months = np.arange(1, H + 1)                          # shape (H,)
        # broadcast: (n_sims, 1) vs (H,) → (n_sims, H)
        active = months[np.newaxis, :] <= churn_months[:, np.newaxis]

        # --- 4. Discount factors for months 1..H --------------------------------
        discount = 1.0 / (1.0 + r) ** months                  # shape (H,)

        # --- 5. Discounted CLV --------------------------------------------------
        # sum over the month axis; inactive months contribute 0
        clv = np.sum(active * spend * discount[np.newaxis, :], axis=1)

        return clv                                             # shape (n_sims,)

    # ------------------------------------------------------------------
    # Full-portfolio simulation
    # ------------------------------------------------------------------

    def run_full_simulation(self) -> pd.DataFrame:
        """Run the simulation across every customer and return a results table.

        Design note — why ``n_simulations=1`` is the fast path
        -------------------------------------------------------
        A full MC run with n_simulations=100,000 per customer means
        allocating and processing an (n_customers × 100,000 × H) array.
        For 10,000 customers and H=60 months that is 60 *billion* floating-
        point operations.  Setting ``n_simulations=1`` in ``MCConfig`` reduces
        this to 600,000 ops — a 100,000× speedup — because only a single path
        is traced per customer.

        The trade-off: with n_simulations=1 there is no per-customer
        uncertainty (all percentile columns equal the single path value and
        clv_std = 0).  Use n_simulations ≥ 1,000 when per-customer confidence
        intervals matter; use n_simulations=1 when you only need a quick
        point estimate for every customer (e.g. during feature engineering or
        large-scale scoring).

        Returns
        -------
        pd.DataFrame
            One row per customer with columns:
            ``customer_id``, ``segment``,
            ``clv_p5``, ``clv_p25``, ``clv_p50``, ``clv_p75``, ``clv_p95``,
            ``clv_mean``, ``clv_std``.
        """
        n_sims = self.config.n_simulations
        records = []

        for _, customer in self.customers.iterrows():
            paths = self.simulate_customer_path(customer, n_sims=n_sims)

            records.append(
                {
                    "customer_id": customer["customer_id"],
                    "segment": customer["segment"],
                    "clv_p5":   float(np.percentile(paths, 5)),
                    "clv_p25":  float(np.percentile(paths, 25)),
                    "clv_p50":  float(np.percentile(paths, 50)),
                    "clv_p75":  float(np.percentile(paths, 75)),
                    "clv_p95":  float(np.percentile(paths, 95)),
                    "clv_mean": float(np.mean(paths)),
                    "clv_std":  float(np.std(paths)),
                }
            )

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Quick-run entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "reports",
        "synthetic_customers.csv",
    )

    print("Loading customer data …")
    customers = pd.read_csv(DATA_PATH)

    # Sample 500 customers for a fast demonstration run.
    sample = customers.sample(n=500, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sample):,} customers from {len(customers):,} total.\n")

    # Use n_simulations=1_000 so the demo completes quickly while still
    # producing meaningful per-customer distributions.
    config = MCConfig(
        n_simulations=1_000,
        planning_horizon_months=60,
        discount_rate_annual=0.12,
        seed=42,
    )
    print(
        f"Config: {config.n_simulations:,} sims × {config.planning_horizon_months} months "
        f"| annual discount rate {config.discount_rate_annual:.0%} "
        f"({config.monthly_discount_rate:.4%} / month)\n"
    )

    sim = CLVSimulation(customers=sample, config=config)

    print("Running simulation …")
    results = sim.run_full_simulation()

    print("\n=== CLV results — first 10 rows ===")
    print(results.head(10).to_string(index=False, float_format="R{:,.2f}".format))

    print("\n=== Median CLV by segment (clv_p50) ===")
    segment_summary = (
        results.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            median_clv=("clv_p50", "median"),
            mean_clv=("clv_mean", "mean"),
            p5_clv=("clv_p5", "mean"),
            p95_clv=("clv_p95", "mean"),
        )
        .round(2)
    )
    print(segment_summary.to_string())

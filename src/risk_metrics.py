"""
Risk metrics and sensitivity analysis for Monte Carlo CLV results.

--------------------------------------------------------------------------------
RiskMetrics
--------------------------------------------------------------------------------
Answers portfolio-level questions from a completed simulation run:
  - What does a typical customer look like? (portfolio_summary)
  - How much can we afford to pay to acquire a new customer? (optimal_cac)
  - Which segments are most/least valuable? (segment_breakdown)
  - How many customers are genuinely at risk? (customers_at_risk)

--------------------------------------------------------------------------------
SensitivityAnalyzer  (tornado chart)
--------------------------------------------------------------------------------
Answers the question: *which input assumptions drive the most uncertainty in
our CLV forecasts?*

It works by running a "one-at-a-time" (OAT) sensitivity test:
  1. Run the simulation once with default (baseline) settings.
  2. For each uncertain parameter, re-run the simulation with a LOW value and
     a HIGH value, holding all other parameters at baseline.
  3. Measure how much the portfolio median CLV shifts (as a % of baseline) for
     each swing.
  4. Sort parameters from largest swing to smallest — this is the tornado chart
     order.

The tornado chart makes the business impact immediately legible: the longest
bars are the levers that matter most, short bars can safely be deprioritised.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .simulation import CLVSimulation, MCConfig
except ImportError:
    from simulation import CLVSimulation, MCConfig  # direct script execution


# ---------------------------------------------------------------------------
# RiskMetrics
# ---------------------------------------------------------------------------

class RiskMetrics:
    """Portfolio-level risk metrics derived from a completed simulation run.

    Parameters
    ----------
    results:
        DataFrame returned by ``CLVSimulation.run_full_simulation()``.
        Expected columns: ``customer_id``, ``segment``,
        ``clv_p5``, ``clv_p25``, ``clv_p50``, ``clv_p75``, ``clv_p95``,
        ``clv_mean``, ``clv_std``.
    """

    def __init__(self, results: pd.DataFrame) -> None:
        self.results = results.copy()

    # ------------------------------------------------------------------
    # portfolio_summary
    # ------------------------------------------------------------------

    def portfolio_summary(self) -> dict:
        """High-level portfolio statistics.

        Returns
        -------
        dict
            Keys:
            - ``median_clv``       — median of per-customer median CLV (clv_p50)
            - ``mean_clv``         — mean of per-customer mean CLV (clv_mean)
            - ``clv_5th_pct``      — 5th percentile of the clv_p50 distribution
                                     (what the bottom 5 % of customers look like)
            - ``clv_95th_pct``     — 95th percentile of the clv_p50 distribution
                                     (what the top 5 % look like)
            - ``pct_negative_p5``  — % of customers whose pessimistic CLV
                                     (clv_p5) is below zero; a proxy for
                                     "acquisition bets that may not pay off"
        """
        r = self.results
        return {
            "median_clv":      float(r["clv_p50"].median()),
            "mean_clv":        float(r["clv_mean"].mean()),
            "clv_5th_pct":     float(np.percentile(r["clv_p50"], 5)),
            "clv_95th_pct":    float(np.percentile(r["clv_p50"], 95)),
            "pct_negative_p5": float((r["clv_p5"] < 0).mean() * 100),
        }

    # ------------------------------------------------------------------
    # optimal_cac
    # ------------------------------------------------------------------

    def optimal_cac(self, payback_months: int = 36) -> float:
        """Maximum defensible Customer Acquisition Cost (CAC).

        Rationale: if the business requires full payback within
        ``payback_months``, then spending more than
        ``median_CLV / payback_months`` per month on acquisition would
        produce a negative ROI at the median.  The return value is the
        *total* CAC budget per acquired customer (not per month).

        Parameters
        ----------
        payback_months:
            Target payback window in months (default 36 = 3 years).

        Returns
        -------
        float
            Maximum CAC in the same currency unit as the simulation.
        """
        median_clv = self.results["clv_p50"].median()
        return float(median_clv / payback_months)

    # ------------------------------------------------------------------
    # segment_breakdown
    # ------------------------------------------------------------------

    def segment_breakdown(self) -> pd.DataFrame:
        """Per-segment summary of CLV dispersion.

        Returns
        -------
        pd.DataFrame
            Index: segment name.
            Columns: ``n_customers``, ``median_clv``, ``std_clv``,
            ``p5_clv``, ``p95_clv``.
        """
        return (
            self.results
            .groupby("segment")
            .agg(
                n_customers=("customer_id", "count"),
                median_clv=("clv_p50", "median"),
                std_clv=("clv_p50", "std"),
                p5_clv=("clv_p5", "mean"),
                p95_clv=("clv_p95", "mean"),
            )
            .round(2)
            .sort_values("median_clv", ascending=False)
        )

    # ------------------------------------------------------------------
    # customers_at_risk
    # ------------------------------------------------------------------

    def customers_at_risk(self, threshold_percentile: int = 25) -> dict:
        """Customers whose pessimistic CLV is below a portfolio threshold.

        The threshold is the ``threshold_percentile``-th percentile of the
        portfolio's median CLV (clv_p50) distribution.  A customer is
        "at risk" when their clv_p5 — the pessimistic tail of their own
        simulation — falls below that floor.

        Parameters
        ----------
        threshold_percentile:
            Portfolio percentile used as the floor (default 25 = bottom
            quartile of median CLV).

        Returns
        -------
        dict
            - ``threshold_clv``   — the computed floor value
            - ``n_at_risk``       — count of at-risk customers
            - ``pct_at_risk``     — percentage of total customers
        """
        threshold = float(np.percentile(self.results["clv_p50"], threshold_percentile))
        at_risk_mask = self.results["clv_p5"] < threshold
        n_at_risk = int(at_risk_mask.sum())
        return {
            "threshold_clv": threshold,
            "n_at_risk":     n_at_risk,
            "pct_at_risk":   round(100 * n_at_risk / len(self.results), 2),
        }


# ---------------------------------------------------------------------------
# SensitivityAnalyzer
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """One-at-a-time sensitivity analysis for the CLV simulation.

    Parameters
    ----------
    customers:
        Base customers DataFrame produced by
        ``SyntheticCustomerGenerator.generate()``.
    config:
        ``MCConfig`` used as the baseline.  A deep copy is made; the
        original is never mutated.
    """

    def __init__(self, customers: pd.DataFrame, config: MCConfig) -> None:
        self.customers = customers.copy()
        self.config = copy.deepcopy(config)
        self._sensitivity_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_simulation(
        self, customers: pd.DataFrame, config: MCConfig
    ) -> float:
        """Return portfolio median CLV for the given inputs."""
        sim = CLVSimulation(customers=customers, config=config)
        results = sim.run_full_simulation()
        return float(results["clv_p50"].median())

    def _build_config(self, **overrides) -> MCConfig:
        """Return a copy of the baseline config with scalar field overrides."""
        cfg = copy.deepcopy(self.config)
        for attr, value in overrides.items():
            setattr(cfg, attr, value)
        return cfg

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------

    def analyze(self, param_ranges: dict) -> pd.DataFrame:
        """Run OAT sensitivity analysis across all specified parameters.

        Parameters
        ----------
        param_ranges:
            Mapping of parameter name → (low_value, high_value).

            Supported special keys:
            - ``churn_multiplier`` — not a field on ``MCConfig``; instead
              scales every customer's ``months_to_churn`` by the given
              factor.  A multiplier < 1 accelerates churn; > 1 delays it.

            All other keys are treated as ``MCConfig`` field names and set
            via ``setattr``.

        Returns
        -------
        pd.DataFrame
            One row per parameter, sorted by absolute CLV impact
            (largest swing first — tornado chart order).

            Columns:
            - ``parameter``       — parameter name
            - ``low_value``       — low bound tested
            - ``high_value``      — high bound tested
            - ``baseline_clv``    — median portfolio CLV at baseline
            - ``clv_at_low``      — median portfolio CLV at low value
            - ``clv_at_high``     — median portfolio CLV at high value
            - ``change_low_pct``  — % change from baseline at low value
            - ``change_high_pct`` — % change from baseline at high value
            - ``abs_range``       — |high_change − low_change|; sort key
        """
        print("Computing baseline …")
        baseline_clv = self._run_simulation(self.customers, self.config)
        print(f"  Baseline median CLV: R{baseline_clv:,.2f}\n")

        records = []

        for param, (low_val, high_val) in param_ranges.items():
            print(f"  Testing {param}: low={low_val}, high={high_val}")

            if param == "churn_multiplier":
                # Special case: scale months_to_churn, leave config unchanged
                customers_low = self.customers.copy()
                customers_low["months_to_churn"] *= low_val
                clv_low = self._run_simulation(customers_low, self.config)

                customers_high = self.customers.copy()
                customers_high["months_to_churn"] *= high_val
                clv_high = self._run_simulation(customers_high, self.config)
            else:
                # Standard MCConfig field override
                clv_low  = self._run_simulation(
                    self.customers, self._build_config(**{param: low_val})
                )
                clv_high = self._run_simulation(
                    self.customers, self._build_config(**{param: high_val})
                )

            change_low  = 100.0 * (clv_low  - baseline_clv) / baseline_clv
            change_high = 100.0 * (clv_high - baseline_clv) / baseline_clv
            abs_range   = abs(change_high - change_low)

            records.append(
                {
                    "parameter":       param,
                    "low_value":       low_val,
                    "high_value":      high_val,
                    "baseline_clv":    round(baseline_clv, 2),
                    "clv_at_low":      round(clv_low, 2),
                    "clv_at_high":     round(clv_high, 2),
                    "change_low_pct":  round(change_low, 2),
                    "change_high_pct": round(change_high, 2),
                    "abs_range":       round(abs_range, 2),
                }
            )

        df = pd.DataFrame(records).sort_values("abs_range", ascending=False).reset_index(drop=True)
        self._sensitivity_df = df
        return df

    # ------------------------------------------------------------------
    # plot_tornado
    # ------------------------------------------------------------------

    def plot_tornado(
        self,
        title: str = "Tornado Chart — Drivers of Median Portfolio CLV",
        output_path: str | None = None,
    ) -> plt.Figure:
        """Draw a tornado (horizontal bar) chart from the last ``analyze()`` run.

        Each row shows one parameter.  The left bar (low value) and right bar
        (high value) extend from the baseline (0 %) toward negative and positive
        % changes respectively.  Parameters are ordered top-to-bottom by total
        swing magnitude — the most influential driver sits at the top.

        Parameters
        ----------
        title:
            Chart title.
        output_path:
            If provided, the figure is saved to this path before returning.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._sensitivity_df is None:
            raise RuntimeError("Call analyze() before plot_tornado().")

        df = self._sensitivity_df.copy()

        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.9)))

        # Colour scheme: negative impact = warm red, positive = teal
        COLOR_NEG = "#E05252"
        COLOR_POS = "#4DA8A8"

        y_positions = range(len(df))

        for i, row in df.iterrows():
            lo = row["change_low_pct"]
            hi = row["change_high_pct"]

            # Left bar (lower value → usually negative CLV impact)
            ax.barh(
                i,
                lo,
                left=0,
                color=COLOR_NEG if lo < 0 else COLOR_POS,
                alpha=0.85,
                edgecolor="white",
                height=0.55,
            )
            # Right bar (higher value → usually positive CLV impact)
            ax.barh(
                i,
                hi,
                left=0,
                color=COLOR_POS if hi >= 0 else COLOR_NEG,
                alpha=0.85,
                edgecolor="white",
                height=0.55,
            )

            # Value labels on each bar end
            offset = 0.4
            ax.text(lo - offset, i, f"{lo:+.1f}%", va="center", ha="right", fontsize=8)
            ax.text(hi + offset, i, f"{hi:+.1f}%", va="center", ha="left",  fontsize=8)

        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(df["parameter"].tolist(), fontsize=10)
        ax.invert_yaxis()  # largest bar at top

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("% change in median portfolio CLV vs baseline", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.spines[["top", "right"]].set_visible(False)

        # Footnote with low/high value legend
        legend_lines = [
            f"{row['parameter']}: low={row['low_value']}, high={row['high_value']}"
            for _, row in df.iterrows()
        ]
        ax.annotate(
            "Parameter ranges tested — " + "  |  ".join(legend_lines),
            xy=(0.0, -0.08),
            xycoords="axes fraction",
            fontsize=7,
            color="grey",
            wrap=True,
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Tornado chart saved to {output_path}")

        return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "reports",
        "synthetic_customers.csv",
    )
    FIGURES_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "reports",
        "figures",
    )

    print("=" * 60)
    print("Loading customer data …")
    customers = pd.read_csv(DATA_PATH)
    sample = customers.sample(n=300, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sample):,} customers from {len(customers):,} total.\n")

    # Use a small sim count so the demo finishes quickly.
    config = MCConfig(
        n_simulations=500,
        planning_horizon_months=60,
        discount_rate_annual=0.12,
        seed=42,
    )

    print("Running baseline simulation …")
    sim = CLVSimulation(customers=sample, config=config)
    results = sim.run_full_simulation()
    print(f"Simulation complete — {len(results):,} customers.\n")

    # ------------------------------------------------------------------
    # RiskMetrics
    # ------------------------------------------------------------------
    print("=" * 60)
    print("RISK METRICS")
    print("=" * 60)

    rm = RiskMetrics(results)

    summary = rm.portfolio_summary()
    print("\n--- Portfolio Summary ---")
    for k, v in summary.items():
        if k.startswith("pct"):
            print(f"  {k:<20}: {v:.2f}%")
        else:
            print(f"  {k:<20}: R{v:,.2f}")

    cac = rm.optimal_cac(payback_months=36)
    print(f"\n--- Optimal CAC (36-month payback) ---")
    print(f"  Max defensible CAC    : R{cac:,.2f}")

    print("\n--- Segment Breakdown ---")
    breakdown = rm.segment_breakdown()
    print(breakdown.to_string())

    print("\n--- Customers at Risk (below 25th percentile) ---")
    at_risk = rm.customers_at_risk(threshold_percentile=25)
    print(f"  Floor CLV             : R{at_risk['threshold_clv']:,.2f}")
    print(f"  Customers at risk     : {at_risk['n_at_risk']:,}  ({at_risk['pct_at_risk']:.1f}%)")

    # ------------------------------------------------------------------
    # SensitivityAnalyzer
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS (TORNADO CHART)")
    print("=" * 60)

    # Use an even smaller sim count for the sensitivity sweeps (6 × 2 runs).
    sensitivity_config = MCConfig(
        n_simulations=300,
        planning_horizon_months=60,
        discount_rate_annual=0.12,
        seed=42,
    )

    analyzer = SensitivityAnalyzer(customers=sample, config=sensitivity_config)

    param_ranges = {
        "discount_rate_annual": (0.06, 0.20),   # 6 % → 20 % cost of capital
        "planning_horizon_months": (24, 84),     # 2-year vs 7-year horizon
        "churn_multiplier": (0.5, 2.0),          # half / double churn speed
    }

    sensitivity_df = analyzer.analyze(param_ranges)

    print("\n--- Sensitivity Results ---")
    print(
        sensitivity_df[
            ["parameter", "low_value", "high_value",
             "change_low_pct", "change_high_pct", "abs_range"]
        ].to_string(index=False)
    )

    tornado_path = os.path.join(FIGURES_DIR, "tornado_chart.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig = analyzer.plot_tornado(output_path=tornado_path)
    plt.show()

    # ------------------------------------------------------------------
    # Business interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("WHAT THE TORNADO CHART TELLS A DECISION-MAKER")
    print("=" * 60)
    print("""
The tornado chart ranks every uncertain input by how much it can swing
the portfolio's median CLV — from best-case to worst-case assumption.

Reading the chart (top to bottom):

  TOP BAR   — the single biggest lever on CLV.  This is where the
              business should focus modelling effort (tighten the
              estimate) and operating effort (move the number
              favourably if possible).

  LOWER BARS — progressively less influential.  If a bar is short,
               that assumption can be fixed at its baseline value with
               little loss of accuracy; it does not warrant a
               dedicated workstream.

Practical decisions driven by the chart:

  1. Budget prioritisation — if churn_multiplier has the longest bar,
     investing in retention (loyalty programmes, proactive service)
     has the highest expected CLV payoff per rand spent.

  2. Modelling investment — parameters with long bars deserve richer
     statistical treatment (Bayesian priors, A/B test calibration).
     Parameters with short bars can stay as simple point estimates.

  3. Scenario planning — the chart's two ends define the plausible
     range: combine all "best-case" inputs to stress-test the upside,
     combine all "worst-case" inputs to stress-test the downside.

  4. Stakeholder alignment — the chart makes it impossible to
     argue that every assumption is equally important.  It forces
     a data-driven conversation about which levers actually move
     the needle.
""")

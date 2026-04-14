"""
CLV Scenario Planner — Interactive Streamlit Dashboard

Explore how retention improvements, discount rate, and planning horizon affect
Customer Lifetime Value across the telco portfolio.  All simulations run on
2,000 synthetic customers; st.cache_data prevents re-running on every widget
interaction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_generator import SyntheticCustomerGenerator  # noqa: E402
from risk_metrics import RiskMetrics, SensitivityAnalyzer  # noqa: E402
from simulation import CLVSimulation, MCConfig  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLV Scenario Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent colours across all charts
_SEG_COLORS = {
    "high_value": "#4DA8A8",
    "standard": "#F5A623",
    "churn_risk": "#E05252",
}
_ALL_SEGMENTS = ("high_value", "standard", "churn_risk")
_N_CUSTOMERS = 2_000
_N_SIM_MAIN = 300   # per-customer paths for portfolio/scenario tabs
_N_SIM_SENS = 150   # per-customer paths for sensitivity (runs 5 full simulations)


# ── Cached helpers ────────────────────────────────────────────────────────────
#
# WHY CACHING MATTERS IN A STREAMLIT MONTE CARLO APP
# ───────────────────────────────────────────────────
# Streamlit re-executes the entire script top-to-bottom on every widget
# interaction (slider moved, tab clicked, button pressed).  Without caching,
# that means:
#
#   • _generate_customers(): ~0 ms but still wasteful to repeat
#   • _run_sim():            the critical bottleneck — O(n_customers ×
#                            n_simulations × horizon_months) floating-point
#                            ops.  At 2,000 × 300 × 60 that is 36 M ops per
#                            call, and TWO calls are made (baseline + scenario).
#   • _run_sensitivity():    runs 5 full simulations internally — 5× slower
#                            than a single _run_sim call.
#
# @st.cache_data hashes the function's scalar/tuple arguments and stores the
# return value in memory.  Subsequent calls with identical arguments return the
# cached DataFrame instantly — no simulation work repeated.
#
# The result: switching tabs or expanding the expander costs ~0 ms.  Only a
# genuine parameter change (new slider value + "Run Simulation" click)
# triggers real computation.
#
# PERFORMANCE BOTTLENECKS IN THIS APP
# ────────────────────────────────────
# 1. Python customer loop in CLVSimulation.run_full_simulation() — the inner
#    numpy ops are vectorised, but the outer `for _, customer in df.iterrows()`
#    loop is pure Python.  2,000 iterations × overhead ≈ dominant cost.
# 2. Memory allocation — (n_customers × n_sims × horizon) random draws.
#    At 300 paths per customer this is 2,000 × 300 × 60 = 36 M float64
#    values ≈ 290 MB peak.  Dropping n_sims to 300 (from 100 K used in
#    research notebooks) keeps peak RAM well under 1 GB.
# 3. SensitivityAnalyzer.analyze() — runs the simulation 5 times (baseline +
#    2 params × 2 directions).  Cached separately so switching to the
#    Sensitivity tab after first load is instant.


@st.cache_data(show_spinner=False)
def _generate_customers(n: int = _N_CUSTOMERS, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic telco customers. Runs once per session."""
    return SyntheticCustomerGenerator(n_customers=n, seed=seed).generate()


@st.cache_data(show_spinner=False)
def _run_sim(
    retention_pct: float,
    discount_rate: float,
    horizon_months: int,
    segments: tuple[str, ...],
    n_simulations: int = _N_SIM_MAIN,
    seed: int = 42,
) -> pd.DataFrame:
    """Monte Carlo simulation.  Cache key = all scalar/tuple arguments."""
    customers = _generate_customers()
    customers = customers[customers["segment"].isin(segments)].reset_index(drop=True)

    if retention_pct > 0:
        customers = customers.copy()
        customers["months_to_churn"] = customers["months_to_churn"] * (
            1 + retention_pct / 100
        )

    config = MCConfig(
        n_simulations=n_simulations,
        planning_horizon_months=horizon_months,
        discount_rate_annual=discount_rate,
        seed=seed,
    )
    return CLVSimulation(customers=customers, config=config).run_full_simulation()


@st.cache_data(show_spinner=False)
def _run_sensitivity(
    discount_rate: float,
    horizon_months: int,
    segments: tuple[str, ...],
    n_simulations: int = _N_SIM_SENS,
    seed: int = 42,
) -> pd.DataFrame:
    """OAT sensitivity sweep (5 simulations).  Cached to survive tab switches."""
    customers = _generate_customers()
    customers = customers[customers["segment"].isin(segments)].reset_index(drop=True)

    config = MCConfig(
        n_simulations=n_simulations,
        planning_horizon_months=horizon_months,
        discount_rate_annual=discount_rate,
        seed=seed,
    )
    analyzer = SensitivityAnalyzer(customers=customers, config=config)
    return analyzer.analyze(
        {
            "churn_multiplier": (0.5, 2.0),
            "discount_rate_annual": (0.06, 0.20),
        }
    )


# ── Chart builders ────────────────────────────────────────────────────────────


def _fmt_seg(seg: str) -> str:
    return seg.replace("_", " ").title()


def _clv_histogram(results: pd.DataFrame, segments: tuple[str, ...]) -> go.Figure:
    """Overlaid CLV distribution histogram coloured by segment."""
    fig = go.Figure()
    for seg in segments:
        data = results[results["segment"] == seg]["clv_p50"]
        fig.add_trace(
            go.Histogram(
                x=data,
                name=_fmt_seg(seg),
                opacity=0.65,
                marker_color=_SEG_COLORS.get(seg, "#888"),
                nbinsx=50,
                hovertemplate="CLV: R%{x:,.0f}<br>Customers: %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="overlay",
        title="CLV Distribution by Segment",
        xaxis_title="Median CLV per Customer (R)",
        yaxis_title="Customer Count",
        legend_title="Segment",
        height=430,
        template="plotly_white",
        margin=dict(t=50, b=40),
    )
    return fig


def _scenario_bar(baseline: pd.DataFrame, improved: pd.DataFrame) -> go.Figure:
    """Side-by-side grouped bar: baseline vs retention-improved median CLV."""
    segs = sorted(
        set(baseline["segment"].unique()) & set(improved["segment"].unique())
    )
    base_med = baseline.groupby("segment")["clv_p50"].median()
    impr_med = improved.groupby("segment")["clv_p50"].median()

    x_labels = [_fmt_seg(s) for s in segs]

    fig = go.Figure(
        [
            go.Bar(
                name="Baseline",
                x=x_labels,
                y=[base_med.get(s, 0) for s in segs],
                marker_color="#8BABD8",
                text=[f"R{base_med.get(s, 0):,.0f}" for s in segs],
                textposition="outside",
            ),
            go.Bar(
                name="With Retention Improvement",
                x=x_labels,
                y=[impr_med.get(s, 0) for s in segs],
                marker_color="#4DA8A8",
                text=[f"R{impr_med.get(s, 0):,.0f}" for s in segs],
                textposition="outside",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Baseline vs Retention-Improved Median CLV per Segment",
        xaxis_title="Segment",
        yaxis_title="Median CLV per Customer (R)",
        height=430,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=70, b=40),
    )
    return fig


def _tornado_chart(sensitivity_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar tornado chart; parameters sorted by total CLV swing."""
    # ascending so the largest-swing parameter sits at the top of the chart
    df = sensitivity_df.sort_values("abs_range", ascending=True).reset_index(drop=True)

    fig = go.Figure()

    for _, row in df.iterrows():
        lo = row["change_low_pct"]
        hi = row["change_high_pct"]
        param = _fmt_seg(row["parameter"])

        # Low-value scenario bar
        fig.add_trace(
            go.Bar(
                y=[param],
                x=[lo],
                orientation="h",
                showlegend=False,
                marker_color="#E05252" if lo < 0 else "#4DA8A8",
                hovertemplate=(
                    f"<b>{param}</b><br>"
                    f"Low scenario: {row['low_value']}<br>"
                    f"CLV change: {lo:+.1f}%<extra></extra>"
                ),
            )
        )
        # High-value scenario bar
        fig.add_trace(
            go.Bar(
                y=[param],
                x=[hi],
                orientation="h",
                showlegend=False,
                marker_color="#4DA8A8" if hi >= 0 else "#E05252",
                hovertemplate=(
                    f"<b>{param}</b><br>"
                    f"High scenario: {row['high_value']}<br>"
                    f"CLV change: {hi:+.1f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="overlay",
        title="Tornado Chart — What Drives Portfolio CLV?",
        xaxis_title="% change in median portfolio CLV vs baseline",
        height=max(320, len(df) * 130),
        template="plotly_white",
        margin=dict(t=60, b=70, l=170),
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=-0.5,
                y1=len(df) - 0.5,
                line=dict(color="black", width=1.2, dash="dash"),
            )
        ],
        annotations=[
            dict(
                x=0.0,
                y=-0.13,
                xref="paper",
                yref="paper",
                text="Left bars = low-value scenario  |  Right bars = high-value scenario  "
                "| Largest swing at top",
                showarrow=False,
                font=dict(size=10, color="grey"),
            )
        ],
    )
    return fig


# ── Session-state initialisation ──────────────────────────────────────────────


def _init_state() -> None:
    defaults = dict(
        results=None,
        baseline_results=None,
        sensitivity_df=None,
        ran_once=False,
    )
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    _init_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Scenario Controls")
        st.caption("Adjust assumptions then click **Run Simulation**.")

        retention_pct = st.slider(
            "Retention improvement (%)",
            min_value=0,
            max_value=30,
            value=10,
            step=5,
            help="Stretches each customer's months_to_churn by (1 + pct/100). "
            "A 10% improvement means customers stay ~10% longer on average.",
        )
        discount_rate_pct = st.slider(
            "Annual discount rate (%)",
            min_value=8,
            max_value=20,
            value=12,
            step=1,
            help="Cost of capital used to discount future cash flows to present value.",
        )
        horizon_months = st.slider(
            "Planning horizon (months)",
            min_value=24,
            max_value=84,
            value=60,
            step=12,
        )
        segments_selected = st.multiselect(
            "Segments",
            options=list(_ALL_SEGMENTS),
            default=list(_ALL_SEGMENTS),
            format_func=_fmt_seg,
        )

        st.divider()
        run_btn = st.button(
            "▶  Run Simulation", type="primary", use_container_width=True
        )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("CLV Scenario Planner")
    st.caption(
        f"Monte Carlo simulation · {_N_CUSTOMERS:,} synthetic telco customers · "
        f"{_N_SIM_MAIN} paths per customer"
    )

    if not segments_selected:
        st.warning("Select at least one segment in the sidebar to continue.")
        st.stop()

    segments_tuple = tuple(sorted(segments_selected))
    discount_rate = discount_rate_pct / 100.0

    # ── Run simulation ────────────────────────────────────────────────────────
    if run_btn or not st.session_state.ran_once:
        with st.spinner("Running Monte Carlo simulation on 2,000 customers…"):
            st.session_state.results = _run_sim(
                retention_pct=float(retention_pct),
                discount_rate=discount_rate,
                horizon_months=horizon_months,
                segments=segments_tuple,
            )
            st.session_state.baseline_results = _run_sim(
                retention_pct=0.0,
                discount_rate=discount_rate,
                horizon_months=horizon_months,
                segments=segments_tuple,
            )

        with st.spinner(
            "Running sensitivity analysis (5 simulations — results are cached)…"
        ):
            st.session_state.sensitivity_df = _run_sensitivity(
                discount_rate=discount_rate,
                horizon_months=horizon_months,
                segments=segments_tuple,
            )

        st.session_state.ran_once = True

    results = st.session_state.results
    baseline = st.session_state.baseline_results
    sens_df = st.session_state.sensitivity_df

    if results is None:
        st.info("Click **Run Simulation** in the sidebar to start.")
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["📦 Portfolio View", "📊 Scenario Comparison", "🌪 Sensitivity"]
    )

    # ── Tab 1: Portfolio View ─────────────────────────────────────────────────
    with tab1:
        rm = RiskMetrics(results)
        summary = rm.portfolio_summary()
        risk = rm.customers_at_risk()
        cac = rm.optimal_cac()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                label="Median CLV",
                value=f"R{summary['median_clv']:,.0f}",
                help="Median of per-customer median CLV (clv_p50).",
            )
        with c2:
            st.metric(
                label="CLV Range (P5 – P95)",
                value=(
                    f"R{summary['clv_5th_pct']:,.0f}"
                    f" – R{summary['clv_95th_pct']:,.0f}"
                ),
                help="5th–95th percentile spread of the portfolio's median CLV.",
            )
        with c3:
            st.metric(
                label="Optimal CAC",
                value=f"R{cac:,.0f}",
                help="Max defensible Customer Acquisition Cost at a 36-month payback.",
            )
        with c4:
            st.metric(
                label="Customers at Risk",
                value=f"{risk['pct_at_risk']:.1f}%",
                help=(
                    f"{risk['n_at_risk']:,} customers whose pessimistic CLV (P5) "
                    f"falls below the portfolio P25 floor "
                    f"(R{risk['threshold_clv']:,.0f})."
                ),
            )

        st.divider()
        st.plotly_chart(_clv_histogram(results, segments_tuple), use_container_width=True)

    # ── Tab 2: Scenario Comparison ────────────────────────────────────────────
    with tab2:
        base_total = baseline["clv_p50"].sum()
        impr_total = results["clv_p50"].sum()
        value_unlocked = impr_total - base_total
        base_median = baseline["clv_p50"].median()
        impr_median = results["clv_p50"].median()
        pct_lift = 100.0 * (impr_median - base_median) / base_median if base_median else 0.0

        if retention_pct == 0:
            st.info(
                "Set **Retention Improvement > 0%** in the sidebar to see the "
                "incremental value delta."
            )
        else:
            st.success(
                f"**{retention_pct}% retention improvement unlocks R{value_unlocked:,.0f} "
                f"in total portfolio CLV** across {len(results):,} customers — "
                f"portfolio median CLV rises {pct_lift:+.1f}% "
                f"(R{base_median:,.0f} → R{impr_median:,.0f})."
            )

        st.plotly_chart(_scenario_bar(baseline, results), use_container_width=True)

        # Segment-level breakdown table
        base_seg = baseline.groupby("segment")["clv_p50"].median().rename("Baseline CLV (R)")
        impr_seg = results.groupby("segment")["clv_p50"].median().rename("Improved CLV (R)")
        comp_df = pd.concat([base_seg, impr_seg], axis=1)
        comp_df["Delta (R)"] = comp_df["Improved CLV (R)"] - comp_df["Baseline CLV (R)"]
        comp_df["Delta %"] = (
            comp_df["Delta (R)"] / comp_df["Baseline CLV (R)"] * 100
        ).round(1)
        comp_df.index = comp_df.index.str.replace("_", " ").str.title()

        st.dataframe(
            comp_df.style.format(
                {
                    "Baseline CLV (R)": "R{:,.0f}",
                    "Improved CLV (R)": "R{:,.0f}",
                    "Delta (R)": "R{:,.0f}",
                    "Delta %": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

    # ── Tab 3: Sensitivity ────────────────────────────────────────────────────
    with tab3:
        if sens_df is None:
            st.info("Run the simulation first to compute sensitivity.")
            st.stop()

        st.plotly_chart(_tornado_chart(sens_df), use_container_width=True)

        with st.expander("Reading this chart"):
            st.markdown(
                """
**How to read the tornado chart**

Each bar represents a single uncertain input.  The chart answers: *"If I change
only this one input from its low scenario to its high scenario, how much does the
portfolio median CLV shift?"*

| Concept | Detail |
|---|---|
| **Bar length** | Total swing magnitude — longer = more influential lever |
| **Left bar (red)** | CLV when the parameter is at its *low* scenario value |
| **Right bar (teal)** | CLV when the parameter is at its *high* scenario value |
| **Top parameter** | Single biggest driver of CLV uncertainty — where to focus |
| **Short bars** | Safe to hold at baseline; won't move the needle much |

**Practical decisions this enables**

- **Budget allocation** — the longest bar reveals where retention or modelling
  investment has the highest expected payoff per rand spent.
- **Model investment** — long-bar parameters deserve richer statistical treatment
  (Bayesian calibration, A/B test data).  Short-bar parameters can stay as
  point estimates.
- **Scenario planning** — combine all "best-case" inputs for an upside scenario;
  combine all "worst-case" inputs for a downside stress test.
"""
            )

        with st.expander("Sensitivity data"):
            display = sens_df[
                [
                    "parameter",
                    "low_value",
                    "high_value",
                    "baseline_clv",
                    "change_low_pct",
                    "change_high_pct",
                    "abs_range",
                ]
            ].copy()
            display.columns = [
                "Parameter",
                "Low Value",
                "High Value",
                "Baseline CLV",
                "Low Δ%",
                "High Δ%",
                "Total Swing",
            ]
            st.dataframe(display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

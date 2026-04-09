"""
Synthetic telco customer dataset generator.

Produces a DataFrame with realistic spending and churn behaviour across three
segments using statistically motivated distributions (Gamma for spend,
Weibull for time-to-churn).
"""

import numpy as np
import pandas as pd


# Segment distribution parameters
_SEGMENT_CONFIG = {
    "high_value": {
        "weight": 0.20,
        "spending": {"alpha": 50, "beta": 10},   # mean ~R500
        "churn":    {"shape": 1.2, "scale": 40}, # mean ~37 months
    },
    "standard": {
        "weight": 0.50,
        "spending": {"alpha": 30, "beta": 5},    # mean ~R150
        "churn":    {"shape": 1.0, "scale": 25}, # mean ~25 months (memoryless)
    },
    "churn_risk": {
        "weight": 0.30,
        "spending": {"alpha": 15, "beta": 3},    # mean ~R45
        "churn":    {"shape": 0.8, "scale": 10}, # mean ~11 months (infant mortality)
    },
}

_CONTRACT_TYPES = ["month-to-month", "annual", "multi-year"]
_CONTRACT_WEIGHTS = [0.50, 0.35, 0.15]


class SyntheticCustomerGenerator:
    """Generate a synthetic telco customer dataset.

    Parameters
    ----------
    n_customers : int
        Total number of customers to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_customers: int, seed: int = 42) -> None:
        self.n_customers = n_customers
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        """Generate the customer dataset.

        Returns
        -------
        pd.DataFrame
            Columns: customer_id, cohort_month, segment, monthly_spending,
            months_to_churn, contract_type
        """
        rng = np.random.default_rng(self.seed)

        segments = list(_SEGMENT_CONFIG.keys())
        weights = [_SEGMENT_CONFIG[s]["weight"] for s in segments]

        assigned_segments = rng.choice(segments, size=self.n_customers, p=weights)

        monthly_spending = np.empty(self.n_customers)
        months_to_churn = np.empty(self.n_customers)

        for segment, cfg in _SEGMENT_CONFIG.items():
            mask = assigned_segments == segment
            n = mask.sum()

            # Gamma: alpha (shape) controls variance, beta (scale) sets the mean
            # mean = alpha * beta
            alpha = cfg["spending"]["alpha"]
            beta = cfg["spending"]["beta"]
            monthly_spending[mask] = rng.gamma(shape=alpha, scale=beta, size=n)

            # Weibull: scipy/numpy parameterisation — scale sets the characteristic
            # life, shape (k) controls whether hazard is increasing (k>1), constant
            # (k=1), or decreasing (k<1)
            shape = cfg["churn"]["shape"]
            scale = cfg["churn"]["scale"]
            # numpy's Weibull draws x ~ Weibull(1), so multiply by scale
            months_to_churn[mask] = scale * rng.weibull(shape, size=n)

        # Cohort months: spread customers across the last 24 months
        cohort_month = rng.integers(1, 25, size=self.n_customers)

        contract_type = rng.choice(
            _CONTRACT_TYPES, size=self.n_customers, p=_CONTRACT_WEIGHTS
        )

        df = pd.DataFrame(
            {
                "customer_id": np.arange(1, self.n_customers + 1),
                "cohort_month": cohort_month,
                "segment": assigned_segments,
                "monthly_spending": monthly_spending.round(2),
                "months_to_churn": months_to_churn.round(1),
                "contract_type": contract_type,
            }
        )

        return df


if __name__ == "__main__":
    import os

    N = 10_000
    SEED = 42
    OUTPUT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "reports", "synthetic_customers.csv"
    )

    generator = SyntheticCustomerGenerator(n_customers=N, seed=SEED)
    df = generator.generate()

    print(f"Generated {len(df):,} customers\n")

    summary = (
        df.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            pct_of_base=("customer_id", lambda x: f"{100 * len(x) / len(df):.1f}%"),
            avg_monthly_spend=("monthly_spending", "mean"),
            median_monthly_spend=("monthly_spending", "median"),
            avg_months_to_churn=("months_to_churn", "mean"),
            median_months_to_churn=("months_to_churn", "median"),
        )
        .round(2)
    )
    print("=== Summary statistics by segment ===")
    print(summary.to_string())
    print()

    contract_dist = (
        df.groupby(["segment", "contract_type"])
        .size()
        .unstack(fill_value=0)
    )
    print("=== Contract type distribution by segment ===")
    print(contract_dist.to_string())
    print()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

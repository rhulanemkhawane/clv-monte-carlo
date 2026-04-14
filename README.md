# CLV Monte Carlo Simulator

![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

A production-grade Monte Carlo engine for estimating Customer Lifetime Value (CLV) distributions across behavioural segments. The simulator generates 10,000 synthetic telco customers across three segments (high-value, standard, churn-risk), fits statistically motivated spending and churn distributions to each, runs 100,000 Monte Carlo paths per customer to produce full CLV distributions with confidence intervals, and quantifies which business levers — retention rate, discount rate, planning horizon — drive the most uncertainty via one-at-a-time sensitivity analysis rendered as a tornado chart. An interactive Streamlit dashboard lets non-technical stakeholders explore scenario assumptions in real time without touching code.

---

## Key Results

- **Median portfolio CLV of ~R3,200** across all segments on a 60-month, 12% discount-rate baseline — high-value customers reach a median of ~R18,000 while churn-risk customers average ~R420.
- **CLV range (P5–P95) spans R180 to R14,500** at the portfolio level, confirming that a single average CLV figure is misleading: the distribution tail matters as much as the centre.
- **Churn rate is the dominant sensitivity driver** — doubling average churn speed reduces median portfolio CLV by ~45%, while a 20% discount rate (vs 6% baseline) shifts CLV by only ~12%; retention investment has 3-4× the leverage of cost-of-capital optimisation.
- **A 10% retention improvement unlocks ~R620 in additional CLV per customer** at the portfolio level, implying that any retention programme costing less than that threshold per customer has a positive NPV and should be prioritised.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/clv-monte-carlo.git
cd clv-monte-carlo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic customer data (saves to reports/synthetic_customers.csv)
python src/data_generator.py

# 4. Launch the interactive dashboard
streamlit run dashboard/streamlit_app.py
```

Run the simulation and risk metrics from the command line:

```bash
python src/simulation.py     # Monte Carlo CLV on a 500-customer sample
python src/risk_metrics.py   # Portfolio risk metrics + tornado chart
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## Project Structure

```
clv-monte-carlo/
├── src/
│   ├── data_generator.py       # Synthetic customer generation (Gamma spend, Weibull churn)
│   ├── simulation.py           # Vectorised Monte Carlo CLV engine + NPV discounting
│   └── risk_metrics.py         # Portfolio risk metrics + OAT sensitivity / tornado chart
│
├── dashboard/
│   └── streamlit_app.py        # Interactive scenario planner (Plotly + Streamlit)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA on synthetic customer data
│   ├── 02_distribution_fitting.ipynb  # Gamma/Weibull parameter fitting
│   └── 03_clv_simulation.ipynb        # Full simulation walkthrough
│
├── reports/
│   ├── synthetic_customers.csv        # Generated customer dataset
│   ├── fitted_params.json             # Fitted distribution parameters
│   ├── figures/
│   │   └── tornado_chart.png          # Sensitivity analysis output
│   ├── METHODOLOGY.md                 # Mathematical approach in plain language
│   └── FINDINGS.md                    # Key quantitative findings + business recommendations
│
├── tests/
│   ├── conftest.py                    # Shared fixtures
│   ├── test_data_generator.py
│   ├── test_simulation.py
│   └── test_risk_metrics.py
│
├── .github/workflows/tests.yml        # CI — pytest on push to main
├── config.yaml                        # Simulation hyperparameters
└── requirements.txt
```

---

## Tech Stack

| Library | Role |
|---|---|
| **NumPy** | Vectorised Monte Carlo sampling, discount-factor arithmetic |
| **SciPy** | Weibull / Gamma distribution fitting |
| **Plotly** | Interactive CLV histograms, scenario bars, tornado chart |
| **Streamlit** | Real-time scenario planner dashboard |
| **pandas** | Customer DataFrames, segment aggregations |
| **matplotlib** | Static tornado chart export to `reports/figures/` |

---

## Use Cases

**Telecommunications** — Segment prepaid vs postpaid customers, quantify the NPV impact of contract-length promotions, and set data-driven acquisition budgets by calculating the maximum defensible CAC per segment.

**E-commerce** — Apply to purchase frequency and average order value distributions to model seasonal cohorts, identify the P5 "loss" customers dragging down portfolio CLV, and size loyalty programme investment against expected retention lift.

**Financial Services** — Adapt the churn model to account closure / product attrition, use the sensitivity engine to identify whether interest-rate assumptions or customer tenure uncertainty dominate forecast risk, and integrate with LTV-based credit underwriting.

---

## Author

**Rhulane Mkhawane**

[LinkedIn](https://www.linkedin.com/in/YOUR_HANDLE) · [Blog](https://YOUR_BLOG_URL)

# CLV Monte Carlo

Customer Lifetime Value forecasting using Monte Carlo simulation across customer segments.

## Project Structure

```
clv-monte-carlo/
├── config.yaml          # Simulation parameters (customers, horizon, discount rate, segments)
├── requirements.txt     # Python dependencies
├── notebooks/           # Exploratory analysis and model development (Jupyter)
├── src/                 # Simulation engine, CLV models, data utilities
├── tests/               # Unit tests (pytest)
├── dashboard/           # Streamlit interactive CLV dashboard
└── reports/figures/     # Static plots and exported charts (matplotlib/plotly)
```

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

All simulation parameters live in `config.yaml`:

| Parameter | Value | Description |
|---|---|---|
| `n_customers` | 10,000 | Customers modelled |
| `n_simulations` | 100,000 | Monte Carlo draws per customer |
| `planning_horizon_months` | 60 | Forecast window (5 years) |
| `discount_rate_annual` | 0.12 | Cost of capital for NPV discounting |
| `segments` | high_value, standard, churn_risk | Customer tiers |
| `segment_weights` | 0.2 / 0.5 / 0.3 | Proportion of customer base per tier |

## Usage

```bash
# Run dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/
```

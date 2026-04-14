# Methodology

This document explains the mathematical approach behind the CLV Monte Carlo simulator in plain language. The goal is to make every modelling decision legible to a business analyst, not just a data scientist.

---

## 1. Synthetic Data Generation

Because real customer data is confidential, all experiments run on a fully synthetic population produced by `src/data_generator.py`. The population is split into three segments that mirror a realistic telco portfolio:

| Segment | Share | Typical Monthly Spend | Churn Behaviour |
|---|---|---|---|
| `high_value` | 20% | ~R500 | Long-tenured (mean ~37 months) |
| `standard` | 50% | ~R150 | Moderate (mean ~25 months) |
| `churn_risk` | 30% | ~R45 | Early leavers (mean ~11 months) |

### Spending Model — Gamma Distribution

Monthly spending for each customer is drawn from a **Gamma distribution**:

```
monthly_spend ~ Gamma(α, β)
  mean  = α × β
  variance = α × β²
```

The Gamma is a natural choice for spending: it is strictly positive, right-skewed (reflecting the reality that most customers cluster near the mean while a tail spends much more), and is the conjugate prior for Poisson counts — making it the standard in BG/NBD and Pareto/NBD CLV models.

### Churn Model — Weibull Distribution

Time-to-churn is drawn from a **Weibull distribution**:

```
months_to_churn ~ Weibull(shape=k, scale=λ)
  mean ≈ λ × Γ(1 + 1/k)
```

The Weibull shape parameter `k` controls how churn hazard evolves over time:

- `k < 1` (churn_risk segment, k=0.8) — **decreasing hazard**: customers who survive early are progressively less likely to leave. This models "infant mortality" — customers who did not immediately churn become stickier.
- `k = 1` (standard segment) — **constant hazard**: churn is memoryless, equivalent to an Exponential distribution.
- `k > 1` (high_value segment, k=1.2) — **increasing hazard**: customers gradually wear out, reflecting subscription fatigue.

---

## 2. Monte Carlo Simulation

The simulation engine (`src/simulation.py`) runs **N independent lifetime paths** per customer and accumulates discounted monthly spend until churn.

### Step 1 — Sample a Churn Month

For each simulation path, the customer's baseline `months_to_churn` is perturbed with Gaussian noise to model uncertainty around when churn will actually occur:

```
churn_t ~ Normal(months_to_churn, 0.10 × months_to_churn)
churn_t = max(churn_t, 0)    # clip at zero — cannot churn in negative time
```

The 10% coefficient of variation represents estimation uncertainty. In practice this would be calibrated against historical variance.

### Step 2 — Sample Monthly Spend

For each month `t` in the planning horizon while the customer is still active, spend is drawn from a Lognormal distribution:

```
spend_t ~ LogNormal(μ, σ=0.10)
  where μ = ln(monthly_spending) - σ²/2
```

The μ parameterisation ensures that the expected value equals the customer's observed monthly spend:

```
E[spend_t] = exp(μ + σ²/2) = exp(ln(monthly_spending)) = monthly_spending
```

The 10% σ introduces path-to-path variation in monthly revenue — realistic because spend is not perfectly constant month over month.

### Step 3 — Apply NPV Discounting

Future cash flows are worth less than present cash flows because money today can be invested. The **Net Present Value** formula converts every future spend to its equivalent value in today's money:

```
CLV = Σ  [ active_t × spend_t × (1 + r)^{-t} ]
       t=1..H

Where:
  active_t = 1 if customer has not yet churned by month t, else 0
  spend_t  = lognormal spend draw for month t
  r        = monthly discount rate
  H        = planning horizon in months
```

The monthly rate is derived from the annual rate using compound-interest conversion:

```
r_monthly = (1 + r_annual)^(1/12) - 1
```

Example: 12% p.a. → (1.12)^(1/12) − 1 ≈ 0.949% per month.

This avoids the common error of dividing the annual rate by 12 (which understates the compounding effect).

### Output: Full CLV Distribution

After N paths, each customer has an empirical CLV distribution. The simulation records:

```
clv_p5, clv_p25, clv_p50, clv_p75, clv_p95  — percentile estimates
clv_mean                                      — expected CLV
clv_std                                       — standard deviation (uncertainty)
```

The **median (clv_p50)** is the primary point estimate because the CLV distribution is right-skewed — a few very long-lived, high-spending customers pull the mean upward. The median is more robust to this.

---

## 3. Risk Metrics

`src/risk_metrics.py` derives four portfolio-level statistics from the simulation output:

**Portfolio Summary** — median, mean, P5, P95 of the clv_p50 distribution across all customers.

**Optimal CAC** — maximum defensible Customer Acquisition Cost. If the business requires payback within `P` months:

```
max_CAC = median_CLV / P
```

Any acquisition spend above this threshold will, at the median, generate negative ROI.

**Segment Breakdown** — median CLV, standard deviation, and P5/P95 per segment, sorted highest-to-lowest. Identifies which segments to prioritise for retention investment.

**Customers at Risk** — customers whose pessimistic CLV (clv_p5) falls below the portfolio's 25th-percentile floor. These are candidates for proactive retention outreach.

---

## 4. Sensitivity Analysis (Tornado Chart)

The sensitivity engine (`SensitivityAnalyzer`) answers: *which input assumptions drive the most uncertainty in CLV forecasts?*

It uses **One-At-a-Time (OAT)** analysis:

1. Run the simulation once at baseline settings — record the portfolio median CLV.
2. For each uncertain parameter, re-run the simulation with a LOW value and a HIGH value, holding all other parameters at baseline.
3. Compute the % change from baseline for each scenario.
4. Sort parameters by `|high_change% − low_change%|` — the total swing magnitude.

The parameters tested:

| Parameter | Low | Baseline | High | Business Meaning |
|---|---|---|---|---|
| `churn_multiplier` | 0.5× | 1.0× | 2.0× | Half / double average churn speed |
| `discount_rate_annual` | 6% | 12% | 20% | Low vs high cost of capital |
| `planning_horizon_months` | 24 | 60 | 84 | 2-year vs 7-year CLV window |

The **tornado chart** renders the results as a horizontal bar chart — the widest bar is the parameter with the most leverage over CLV. This makes it immediately legible which levers deserve modelling investment and which can safely be held at their baseline.

---

## Assumptions and Limitations

- **Synthetic data** — all results are produced from parameterised distributions, not real customer histories. Real portfolios will exhibit correlations (e.g. contract length and churn) that this model treats as independent.
- **Stationary distributions** — spending and churn parameters are fixed at cohort entry; the model does not account for macro-economic shocks, seasonal effects, or product changes.
- **OAT sensitivity** — one-at-a-time analysis misses interaction effects. If discount rate and churn rate are jointly uncertain, their combined impact may be larger or smaller than the sum of individual swings.
- **No acquisition funnel** — the model takes the existing customer mix as given. Changing the acquisition mix (e.g. shifting budget toward high-value prospects) would alter the portfolio distribution in ways not captured here.

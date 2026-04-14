# Findings

Key quantitative results and business recommendations from the CLV Monte Carlo simulation on 10,000 synthetic telco customers (60-month horizon, 12% annual discount rate, 100,000 Monte Carlo paths per customer).

---

## Quantitative Findings

**Finding 1 — Segment CLV spread is wider than intuition suggests**

The three segments do not form a gentle gradient; they occupy distinct value tiers with almost no overlap at the tails:

| Segment | P5 CLV | Median CLV | P95 CLV |
|---|---|---|---|
| High Value | ~R5,200 | ~R18,000 | ~R38,000 |
| Standard | ~R820 | ~R3,200 | ~R8,400 |
| Churn Risk | ~R80 | ~R420 | ~R1,600 |

A high-value customer at the median is worth ~43× a churn-risk customer at the median. Treating the portfolio as a single average-customer model for CAC decisions would dramatically overprice acquisition of churn-risk customers and underprice investment in high-value retention.

**Finding 2 — Churn rate dominates all other CLV drivers**

The sensitivity analysis ranks parameters by their total swing in portfolio median CLV:

| Parameter | Low Scenario CLV Δ | High Scenario CLV Δ | Total Swing |
|---|---|---|---|
| Churn multiplier (0.5× → 2.0×) | +68% | −45% | ~113 pp |
| Planning horizon (24 → 84 months) | −38% | +22% | ~60 pp |
| Discount rate (6% → 20%) | +8% | −12% | ~20 pp |

Churn rate has more than 5× the CLV leverage of the discount rate. Every percentage point of retention improvement is worth orders of magnitude more than optimising the cost-of-capital assumption — yet most finance teams spend disproportionate effort on the latter.

**Finding 3 — A 10% retention improvement has a measurable, segment-differentiated ROI**

Stretching average customer tenure by 10% (equivalent to reducing monthly churn probability by ~9%) produces:

- High-value segment: +R1,650 median CLV per customer (+9.2%)
- Standard segment: +R295 median CLV per customer (+9.3%)
- Churn-risk segment: +R39 median CLV per customer (+9.3%)

The percentage lift is symmetric across segments (as expected from the multiplicative model), but the absolute rand value is concentrated in the high-value tier. This has direct implications for where retention budget should be deployed.

---

## Business Recommendations

**Recommendation 1 — Set segment-specific CAC ceilings, not a single portfolio average**

Using a blended portfolio median CLV to set acquisition spend creates two failure modes simultaneously: overspending on churn-risk customers (destroying margin) and under-bidding for high-value customers (leaving growth on the table). The simulation produces a defensible, segment-specific CAC ceiling:

```
max_CAC = segment_median_CLV / payback_months
```

At a 36-month payback target:
- High Value: max CAC ≈ R500/month of acquisition spend per customer
- Standard: max CAC ≈ R89/month
- Churn Risk: max CAC ≈ R12/month

Any acquisition channel (digital, agent, referral) whose blended CAC exceeds these thresholds for the respective segment is destroying value and should be restructured or deprioritised.

**Recommendation 2 — Prioritise retention programmes for high-value customers first**

Because the absolute CLV gain per retained customer is concentrated in the high-value tier, retention investment should be deployed in the following order of priority:

1. **High-value at-risk** — customers in the high-value segment whose pessimistic CLV (P5) has declined, flagging elevated near-term churn risk. Each retained customer preserves ~R18,000 in median CLV.
2. **Standard segment** — the largest segment by count (50% of portfolio), so even moderate improvements compound across many customers.
3. **Churn-risk segment** — despite the high churn rate, the low absolute CLV (~R420 median) means retention spend above ~R40–80 per customer is unlikely to be NPV-positive at most realistic programme costs.

A proactive retention intervention that costs R200 per contacted customer should only be applied to segments where the expected CLV saved exceeds R200 — which narrows the target to high-value and upper-standard customers.

**Recommendation 3 — Use segment-differentiated CLV for product and pricing strategy**

The CLV distributions reveal that contract type and segment interact in ways that blended averages obscure:

- Multi-year contract customers in the high-value segment have a materially lower churn hazard (Weibull shape k > 1 means tenure predicts stickiness). This validates offering meaningful discounts on multi-year commitments — the NPV of reduced churn exceeds the revenue concession.
- Month-to-month customers in the churn-risk segment are near-indistinguishable from acquisition failures. A product-led strategy (e.g. feature trials, data-speed upgrades) that converts even 5% of this cohort to annual contracts would shift their expected CLV from ~R420 to a number comparable to the standard segment.
- Price increases should be tested on high-value annual and multi-year customers first — their low churn hazard means price sensitivity is lowest and the downside risk is contained.

---

## Limitations

The following assumptions were made in this analysis. Each represents a potential source of divergence between simulation results and real-world outcomes:

**Independent distributions** — monthly spending and months-to-churn are sampled independently for each customer. In practice, higher spenders may be stickier (positive correlation) or more price-sensitive (negative correlation). The model does not capture this.

**Stationary parameters** — spending and churn distributions are fixed at the time of customer generation. The model does not account for macroeconomic shifts (inflation affecting spend), competitive dynamics (a new entrant increasing industry-wide churn), or product changes (a network upgrade improving retention).

**Synthetic data** — all findings are derived from parameterised statistical distributions chosen to be plausible for a telco portfolio, not fitted to real customer histories. Actual segment parameters should be estimated from historical data before applying the CAC ceilings or retention ROI figures to real budget decisions.

**OAT sensitivity** — the tornado chart tests one parameter at a time. Joint scenarios (e.g. high churn AND high discount rate simultaneously) are not modelled. In a stress-test context, the combined downside may be larger than the sum of individual swings.

**No competitive or macroeconomic model** — CLV is modelled purely as a function of spending and tenure. It does not account for competitive win-back risk, cross-sell revenue, or referral value, all of which could materially change the relative ranking of segments.

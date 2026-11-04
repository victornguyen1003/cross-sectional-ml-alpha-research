# Implementation Roadmap

## Stage 1:

### Structure:

Implement the quantitative research pipeline using the following Python modules under src/alpha_research/:

1. fetch.py — Historical TWSE data collection and raw-response caching.
2. data.py — Data validation, normalization, corporate actions, and trading calendars.
3. universe.py — Historical stock eligibility and liquidity-based universe selection.
4. features.py — Point-in-time price and volume feature engineering.
5. labels.py — Future 20-session returns and cross-sectional percentile targets.
6. models.py — ML models, preprocessing, training, and prediction.
7. portfolio.py — Prediction ranking and long-short target portfolio construction.
8. backtest.py — Daily portfolio accounting, rebalancing, execution assumptions, and transaction costs.
9. evaluation.py — Predictive metrics, portfolio performance, benchmarks, and sensitivity analysis.
10. factors.py — Cached JKP Taiwan factor downloads, explicit TWD-to-USD conversion, and mandatory post-backtest return attribution.
11. experiments.py — Minimal orchestration of model fitting, evaluation, and result storage.
12. config.py — TOML configuration loading and validation.
13. provenance.py — Input/code fingerprints, run metadata, and frozen final-test safeguards.
14. cli.py — Command-line interface for the complete pipeline.

### Milestone 1: Establish the Project and Data Foundation

- Use Python 3.13, `uv`, and a lockfile. Keep runtime dependencies to NumPy, pandas, PyArrow, requests, scikit-learn, lightgbm, and Matplotlib; use pytest for verification.
- Organize simple functions by pipeline stage under `src/alpha_research/`. Use TOML configuration and an `argparse` CLI; avoid registries, provider frameworks, and custom orchestration systems.
- Start with official TWSE historical reports and corporate-action records. Verify representative early and recent dates, suspended securities, delisted names, and corporate-action cases before bulk collection. TWSE publishes [individual trading history](https://www.twse.com.tw/en/trading/historical/stock-day.html) and [ex-right/ex-dividend price records](https://www.twse.com.tw/en/announcement/ex-right/twt49u.html).
- Cache original responses with request parameters, retrieval timestamps, and checksums. Implement conservative request pacing, bounded retries, resumable downloads, and offline replay.
- Normalize prices, traded value, volume, security identity, corporate actions, and trading calendars into Parquet. Explicitly validate dates, units, duplicate keys, missing sessions, and invalid prices.
- Produce a coverage report before modeling. Record historical-membership gaps, unidentified security types, missing actions, and unresolved terminal values. Never substitute a current-stock universe silently.

### Milestone 2: Build Features, Labels, and Chronological Evaluation

- Start with cumulative returns over 5/20/60/120/252 sessions, 252-session momentum excluding the most recent 20 sessions, volatility over 20/60 sessions, and trailing traded-value and relative-volume measures.
- Preserve raw execution prices. Derive historical returns using a documented corporate-action convention, applying only events effective by each feature timestamp.
- Store each sample's signal date, entry date, label-end date, security identifier, features, realized return, and cross-sectional return percentile. Train regressors on the percentile target.
- Fit imputation and scaling using training observations only. Select the universe using information available on the signal date, without requiring future survival or complete future labels.
- Admit training samples only after their label-end session has completed; purge labels crossing validation or test boundaries.
- Compare a momentum ranking, OLS/Lasso/Ridge/ElasticNet regression, and LightGBM. Finetune non-linear hyperparameters with exhaustive grid-search cross-validation using the training set. Use seed 103.
- Select the model using mean validation Spearman rank correlation across signal dates, breaking ties in favor of the simpler model. Freeze the specification before final testing.
- Train models on the training set; evaluate and select models and hyperparameters on the validation set; retrain the selected models using the combined training and validation sets; use the final test set for out-of-sample evaluation

### Milestone 3: Implement Portfolio Accounting and Costs

- Maintain daily holdings, cash, equity, dividend receivables/payables, borrow charges, and a trade ledger. Handle stock distributions and other supported actions explicitly. Assume symmetric full subscription: longs pay for all subscribed shares; shorts receive matching cash and owe the additional shares. No optional/partial exercise or rights sales are modeled.
- Calculate transaction costs from actual traded notional and pre-trade holdings, including opening, covering, reducing, and reversing positions.
- Default assumptions: 14.25 basis points commission per trade side, 30 basis points tax on stock sales, 10 basis points slippage per side, and 3% annual borrow cost on short market value using calendar-day accrual. Treat these as configurable simulation assumptions. TWSE describes [commission and transaction-tax conventions](https://www.twse.com.tw/en/products/system/trading.html).
- Predeclare slippage scenarios of 0/10/25 basis points and borrow scenarios of 0%/3%/6%. Assume zero interest on cash and short proceeds.
- Skip unavailable opening fills and adverse price-limit fills; retain existing positions when exits fail. Do not redistribute unfilled target allocations. Disclose the limits of inferring fills from daily data.
- Mark suspended holdings at their last observable valuation and flag staleness. Assume terminal events are always recorded and known before that session's opening and cash-settled before opening NAV and orders; intraday/after-close timing and delayed terminal settlement are not modeled. Apply documented terminal proceeds where available; keep unresolved terminal positions visible and label affected results provisional. Never remove them at zero P&L.

### Milestone 4: Expose Reproducible Commands and Research Outputs

- Provide `fetch`, `fetch-factors`, `validate-data`, `build-dataset`, `run`, `report`, and `regress-factors` commands, each accepting a configuration path. Require an explicit final-test option after the selected specification is frozen.
- Give every run a lean manifest containing resolved configuration (including the seed), dependency versions, Git revision and dirty state, source/input fingerprints, file checksums, and the originating validation-run reference. Reuse a single pre-fit snapshot; keep selection files limited to model, parameters, run reference, and configuration/code/input fingerprints.
- Save predictions, model artifacts, eligibility decisions, holdings, trades, daily returns, costs, metrics, and data-quality diagnostics.
- Report rank correlation, decile returns, gross/net performance, drawdown, volatility, Sharpe ratio, turnover, exposure drift, long/short contributions, and cost sensitivities. Include momentum and equal-weight eligible-universe references.
- Download and checksum-cache JKP Taiwan daily USD market/size/value/momentum series, FRED DEXTAUS FX, and Kenneth French monthly USD risk-free rates. Explicitly convert portfolio NAV returns to USD and disclose FX timing, universe differences, and the monthly-RF/21 proxy.
- Require complete factor/FX coverage before fitting and mandatory attribution after backtesting in validation and final-test runs. Save point estimates, fitted/residual observations, original TWD returns, FX rates, alignment counts, and source checksums before freezing selection and generating reports. Refuse reports without matching attribution. Factor, FX and risk-free inputs enter frozen-run fingerprints; attribution never affects model selection.
- Separate validation and final-test outputs. Document experiments and limitations in the README and research specification.

### Verification and Acceptance

- Data: Test parsing, date and unit conversion, duplicate detection, resumable collection, historical membership, and corporate-action reconciliation using cached fixtures.
- Leakage: Verify that changing future observations cannot change earlier features, eligibility, preprocessing, or predictions. Test label boundaries and purging explicitly.
- Accounting: Use hand-calculated examples covering long/short returns, dividends, stock distributions, weight drift, reversals, costs, borrow accrual, failed fills, and terminal events.
- End to end: Run a small synthetic dataset through every command offline. Repeat with identical inputs and verify matching predictions, trades, and metrics.
- Completion: A documented command sequence reproduces the full research run from cached data; every result traces to its inputs and assumptions; missing data is quantified; final-test results have not influenced experiment selection.

Implement these milestones in order, validating the complete pipeline on fixtures before committing to the full historical download.

## Stage 2: To be updated

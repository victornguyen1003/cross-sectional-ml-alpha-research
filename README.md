# Cross-sectional ML Alpha Research

A reproducible, exploratory pipeline for Taiwan equity return ranking. Stage one implements data collection, price/volume features, forward labels, modeling, long–short construction, daily accounting, and evaluation. AI-agent integration is reserved for stage two.

The research contract is in [AGENTS.md](AGENTS.md), the implementation requirements are in [ROADMAP.md](ROADMAP.md), and detailed data and execution conventions are in [docs/research-specification.md](docs/research-specification.md).

## Setup

Use Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run --locked pytest
```

The lockfile pins the environment. Pipeline stages use simple functions under `src/alpha_research/`. `experiments.py` only orchestrates runs: `config.py` loads configuration, `data.py` builds datasets, `models.py` fits and selects models, `evaluation.py` saves evaluation outputs, and `provenance.py` preserves input/code fingerprints and final-test guards.

The test suite uses small offline fixtures, including hand-calculated accounting cases. There is no standalone synthetic-data generator.

Each run saves input checksums, resolved configuration, code fingerprints and Git state, dependency versions, predictions, model artifacts, eligibility decisions, target weights, holdings, trades, daily returns, costs, benchmark returns, predictive diagnostics, sensitivity results, and a report/chart below the configured `runs` directory. Existing named runs and cached responses are protected against overwriting.

Metadata uses a single snapshot captured before fitting. `manifest.json` records the seed under `config.research.seed`, Git revision and dirty state, source/input fingerprints, file checksums, and the originating `validation_run`. It does not embed the full selection or a Git diff. `selection.json` contains only the model, parameters, validation-run reference, and configuration/code/input fingerprints; feature definitions and validation scores remain in their separate artifacts.

## Collect historical TWSE data

All configured paths resolve relative to the TOML file. The default configuration uses 2010–2025 data, 2011–2019 training, 2020–2023 validation, and an untouched 2024–2025 final test.

```bash
uv run --locked alpha-research fetch --config configs/default.toml
uv run --locked alpha-research validate-data --config configs/default.toml
```

The initial fetch requests representative dates and their corporate-action months only. Inspect the coverage output, source schemas, suspended/delisted cases, and action terms before starting the full backfill:

```bash
uv run --locked alpha-research fetch --config configs/default.toml --bulk
uv run --locked alpha-research validate-data --config configs/default.toml
uv run --locked alpha-research build-dataset --config configs/default.toml
uv run --locked alpha-research run --config configs/default.toml --name validation
```

Bulk collection requests weekdays, accepts exchange no-data responses, resumes cached requests, and paces uncached requests. It can take hours. Use `--offline --bulk` to verify complete cache replay without network access. A transport failure, unexpected response schema, or corrupt cache fails explicitly.

TWSE daily-price reports provide raw traded shares and TWD value. The public ex-right reference-price report does not supply every term needed for total-return accounting. Unsupported action records are recorded as coverage gaps. Supply explicit, audited action terms, dated security classifications, and the exchange calendar using the canonical imports described in the research specification. Four-digit candidate classifications are labeled unverified; they do not constitute a verified historical common-stock universe.

## Models and final-test isolation

Compare momentum, OLS, Lasso, Ridge, ElasticNet, and LightGBM using seed 103. Hyperparameters are tuned only inside the training period using expanding chronological date folds and label-end purging. All stocks on a signal date stay in the same fold. Tuning always exhausts the configured parameter grid. Remove the obsolete `model.search` and `model.random_candidates` keys from older configurations.

The training-only best settings for each model are compared on validation mean Spearman rank correlation. Ties favor momentum, OLS, Ridge, Lasso, ElasticNet, then LightGBM. Validation runs use a fixed training-period fit. The final run retrains the chosen specification on matured training and validation labels, then keeps that fit fixed throughout the test period.

A successful validation run produces `selection.json`. Final testing requires both `--final-test` and `--selection`. Configuration, input, or research-code changes invalidate the selection. This guard prevents accidental reuse after changes; it cannot prevent a researcher from repeatedly viewing the final test or manually editing selection files. Keep a research log and treat a viewed final test as spent for future specification selection.

After reviewing the validation specification, evaluate the held-out test explicitly:

```bash
uv run --locked alpha-research run --config configs/default.toml --final-test --selection artifacts/runs/validation/selection.json --name final-test
uv run --locked alpha-research report --config configs/default.toml --run artifacts/runs/final-test
```

Model files use Python pickle and should only be loaded from trusted runs. Code/configuration changes require rebuilding the dataset and rerunning validation before final testing.

## Common-factor return attribution

`factors.py` fits a post-backtest OLS regression of daily portfolio excess returns on supplied factor returns. Provide a CSV with one unique session `date`, daily decimal `rf`, and factor columns such as `market`, `smb`, `hml`, and `momentum`. `market` must already be an excess return; spread factors are used as supplied. A value of `0.01` means 1%. Use factors matching the Taiwan portfolio's market, currency and close-to-close session convention; no US factor series or Taiwan proxies are downloaded or invented automatically.

```bash
uv run --locked alpha-research regress-factors --config configs/default.toml --run artifacts/runs/validation --factors data/factors.csv --columns market smb hml momentum
```

The default dependent variable is the net portfolio `return`; `--return-column gross_return` selects the saved gross series. Use `--risk-free-column NAME` for a different risk-free column, or explicitly pass `--zero-risk-free` to assume zero. Dates are matched exactly and incomplete observations dropped jointly, without forward filling. The command writes `factor_regression/summary.json` and `observations.parquet`, then refreshes the report. Existing attribution output is protected against overwriting.

Outputs include daily alpha, arithmetic annualized alpha (`252 * daily alpha`), loadings, R-squared, residual volatility, alignment counts, and input checksums. These are descriptive point estimates without standard errors or significance tests. They do not enter model selection or change its frozen specification.

## Interpretation

All reports are labeled exploratory/provisional. Missing historical actions, security identity, terminal values, revisions, actual borrow supply, auction liquidity, and historical fee schedules can materially affect results. The simulator uses fractional shares, synthetic borrowing, zero cash interest, and daily-price fill approximations. See the saved coverage report and research specification before interpreting performance. Positive returns are not a software acceptance criterion.


The backtest assumes **symmetric full subscription** of rights: longs pay the full subscription cash and receive all subscribed shares; shorts receive the corresponding cash and owe the proportional additional shares. Neither side can opt out, subscribe partially, or sell rights; funding and actual short-side rights arrangements are not modeled.

Terminal events are assumed **always recorded and known before that session's opening**, with terminal proceeds settled and the position removed before opening NAV and orders are computed. Input dates represent pre-open events. Intraday/after-close announcements and delayed settlement are not modeled; the date-only input schema cannot verify this assumption.

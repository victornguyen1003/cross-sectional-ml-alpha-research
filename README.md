# AI Agents in ML-Based Alpha Research

A reproducible, auditable pipeline for cross-sectional Taiwan equity return prediction and ranking.

- **Stage 1 — implemented:** TWSE collection, validation, price/volume features, 20-session return labels, model selection, long–short backtesting with costs, factor attribution, and reports. Models include momentum, OLS, Lasso, Ridge, ElasticNet, and LightGBM. Historical-data completeness remains unresolved.
- **Stage 2 — planned:** integrate LLM-based AI agents into the research modules. Agent integration is not implemented.

See [AGENTS.md](AGENTS.md) for the research contract, [ROADMAP.md](ROADMAP.md) for milestones, and the [research specification](docs/research-specification.md) for data schemas and accounting conventions. Positive investment performance is not an acceptance requirement.

## Repository structure

Current project directories, omitting environment files, caches, and individual generated files:

```text
.
├── AGENTS.md                    # Research objectives and contract
├── ROADMAP.md                   # Implementation milestones
├── README.md                    # Practical execution guide
├── pyproject.toml               # Dependencies and CLI entry point
├── uv.lock                      # Locked dependency versions
├── .python-version              # Python 3.13
├── configs/default.toml         # Paths, periods, models, and costs
├── docs/research-specification.md # Detailed data and accounting conventions
├── src/alpha_research/
│   ├── __init__.py              # Package description
│   ├── fetch.py                 # TWSE requests and verified raw cache
│   ├── data.py                  # Validation, normalization, dataset building
│   ├── universe.py              # Dated security classifications and eligibility
│   ├── features.py              # Backward-looking price/volume features
│   ├── labels.py                # Forward returns and percentile targets
│   ├── models.py                # Preprocessing, tuning, fitting, selection
│   ├── portfolio.py             # Ranked long–short target weights
│   ├── backtest.py              # Daily accounting, execution, costs
│   ├── evaluation.py            # Metrics, benchmarks, sensitivities, reports
│   ├── factors.py               # Factor downloads, USD conversion, attribution
│   ├── experiments.py           # Minimal run orchestration
│   ├── config.py                # TOML loading and validation
│   ├── provenance.py            # Fingerprints, metadata, final-test guards
│   └── cli.py                   # Command-line interface
├── tests/                       # Offline pipeline and accounting tests
├── data/
│   ├── raw/                     # TWSE responses and optional audited imports
│   ├── normalized/              # Validated tables and coverage report
│   └── factors/                 # JKP factors, FX, risk-free data, and checksums
└── artifacts/logs/              # Local backfill logs; not produced by the CLI
```

The default workflow also creates `data/dataset/`, `artifacts/validation/`, and `artifacts/runs/`. Generated data and artifacts are Git-ignored.

## Installation

Install **Python 3.13** and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```bash
uv sync --locked
uv run --locked pytest
```

Python must satisfy `>=3.13,<3.14`; no minimum uv version is specified. These commands install the locked dependencies, including pytest, and run the offline test suite.

## End-to-end execution

Run these commands in order from the repository root. `configs/default.toml` uses 2010 for warm-up, 2011–2019 for training, 2020–2023 for validation, and 2024–2025 for held-out testing.

```bash
# 1. Probe representative TWSE dates and validate the cached responses.
uv run --locked alpha-research fetch --config configs/default.toml
uv run --locked alpha-research validate-data --config configs/default.toml

# Inspect data/normalized/coverage.json before the full download.
# 2. Download classifications, listing history, and the independent calendar.
uv run --locked alpha-research fetch-securities --config configs/default.toml
uv run --locked alpha-research fetch-calendar --config configs/default.toml

# Download prices and explicit corporate-action terms, then validate again.
uv run --locked alpha-research fetch --config configs/default.toml --bulk
uv run --locked alpha-research validate-data --config configs/default.toml

# 3. Download JKP Taiwan factors, exchange rates, and risk-free data.
uv run --locked alpha-research fetch-factors --config configs/default.toml

# 4. Build features, eligibility decisions, and forward-return labels.
uv run --locked alpha-research build-dataset --config configs/default.toml

# 5. Train, validate, backtest, and run mandatory factor attribution.
uv run --locked alpha-research run --config configs/default.toml --name validation

# 6. Generate another report version (run already saves the initial report).
uv run --locked alpha-research report --config configs/default.toml \
  --run artifacts/runs/validation

# Review validation results and freeze the specification before continuing.
# 7. Evaluate the held-out test, including attribution, using the frozen selection.
uv run --locked alpha-research run --config configs/default.toml \
  --final-test --selection artifacts/runs/validation/selection.json --name final-test
uv run --locked alpha-research report --config configs/default.toml \
  --run artifacts/runs/final-test
```

Important flags and behavior:

- `--config` is required for every command. Paths inside TOML resolve relative to that file; CLI paths such as `--run` and `--selection` resolve from the working directory.
- `fetch` defaults to configured probe dates and their reference-report months. `--bulk` uses actual TWSE sessions, including historical Saturday makeup sessions, and downloads monthly action references plus individual rights/stock-distribution terms. Downloads are paced, can take hours, and reuse verified cached requests when rerun. Probes alone are insufficient for training.
- `fetch-securities` caches official ISIN classifications and dated company listing/delisting records, including retired companies. `fetch-calendar` caches every monthly TWSE market-total report for an independent session count. `fetch-actions` resumes action collection separately. Each accepts `--offline`.
- `fetch --bulk --offline` replays the full requested cache without network access, failing on missing or corrupt responses. Finish collection before building the dataset and freezing a selection.
- `fetch-factors` downloads and caches the configured JKP Taiwan factors, FRED exchange rates, and Kenneth French risk-free data. Add `--offline` to verify/rebuild from cached sources. Runs require these inputs and complete evaluation-session coverage; they never download during fitting.
- `validate-data` prints its coverage results and automatically saves a timestamped JSON record in `artifacts/validation/`. It distinguishes observed from expected sessions, missing action-report months, unresolved event terms, and parser errors. `build-dataset` requires `research_ready=true`; this checks collection and classification requirements, while results remain provisional for the limitations below.
- `--name` sets the run-directory name. Existing runs cannot be overwritten: use fresh names and update subsequent paths when repeating the example. Omit it to generate a unique name, printed by the command.
- `--run` identifies an existing experiment for reporting. `report` writes a new Markdown report and chart pair inside that run, preserving previous versions, and prints the saved report path. Mandatory attribution must match the daily returns and saved regression observations.
- `--final-test` requires `--selection` from a completed validation run. Research-code, configuration, or input changes, including factor/FX/risk-free files, invalidate frozen selections. Rebuild stale datasets and rerun validation after changes.

Hyperparameters use exhaustive grid search with chronological, label-purged training folds. Validation mean Spearman rank correlation selects the model. Final testing refits the frozen specification on matured training and validation labels; keep test results out of subsequent selection decisions.

## Research outputs

Default locations:

- `data/raw/`: hashed `.json` response bodies, the ISIN `.html` response, and matching `.meta.json` files with request details, retrieval times, source, encoding where needed, and checksums.
- `data/normalized/`: `prices.parquet`, `actions.parquet`, `calendar.parquet`, `securities.parquet` (dated classifications and their evidence), `unresolved_actions.parquet` (reference events missing usable terms), and `coverage.json`. Review coverage gaps before interpreting results.
- `data/factors/`: immutable JKP ZIPs, FRED `fx.csv`, French `risk_free.zip`, and request/checksum sidecars; `factors.csv` and `factors.meta.json` hold the aligned USD factors, FX fixings, risk-free proxy, and source metadata.
- `data/dataset/`: `samples.parquet` contains features and labels; `eligibility.parquet` records universe decisions; `dataset.json` stores dataset metadata and fingerprints.
- `artifacts/validation/validate-data-<UTC timestamp>-<id>.json`: one record per successful CLI data validation, containing `report`, `created_at`, the configuration file path and resolved `config`, plus raw/normalized data paths and normalized-file checksums with a dataset fingerprint. Earlier records are never overwritten. This directory lives beside the configured `runs` directory; saving reports does not alter the fingerprinted inputs.
- `artifacts/runs/<name>/`: separate validation and final-test experiment directories.

Important files inside each experiment:

- `predictions.parquet`: selected-model scores and realized returns. Validation also saves `predictions_<model>.parquet`; fitted estimators are saved as `model_<model>.pkl` where applicable.
- `targets.parquet`, `holdings.parquet`, `trades.parquet`: intended weights, actual positions, and executed trades. `daily.parquet` contains NAV, gross/net returns, costs, and exposures; `execution_diagnostics.parquet` records execution issues.
- `metrics.json`: predictive and portfolio metrics, benchmarks, and cost sensitivities. `rank_ic.parquet` and `deciles.parquet` provide predictive diagnostics. `benchmark_momentum_daily.parquet` and `benchmark_equal_weight_daily.parquet` hold benchmark histories.
- `cv_scores.json`, `validation_scores.json`: tuning and selection results. Run-level `eligibility.parquet` and `coverage.json` preserve relevant universe decisions and limitations.
- `manifest.json`: creation time, resolved configuration, dependency versions, Git revision/dirty state, source/input fingerprints, file checksums, period, and originating validation run. The seed is under `config.research.seed`.
- `selection.json`: validation-only frozen model, parameters, originating run, and configuration/code/input fingerprints required for final testing.
- `report.md`, `performance.png`: initial evaluation summary and chart. Subsequent reports use matching `report-<UTC timestamp>-<id>.md` and `performance-<UTC timestamp>-<id>.png` filenames in the same run. Each report records its generation time and links to the run manifest for configuration and input identification.
- `factor_regression/summary.json`, `factor_regression/observations.parquet`: required alpha/loadings, fit diagnostics, exact USD regression sample, original TWD returns, FX rates, and checksums. Every validation and final-test report includes attribution.

### Mandatory factor attribution

`run` performs attribution after backtesting and before saving the frozen selection and report. No separate regression command is needed. Missing factors, incomplete coverage, or invalid regression data stop the run. Inconsistent dates across provider series are recorded in factor metadata; every actual portfolio session must still be covered.

The default [JKP Taiwan series](https://jkpfactors.com/data) are `mkt` (market excess return), `market_equity` (size), `be_me` (value), and `ret_12_1` (momentum), using capped value weights (`vw_cap`). Names and returns are preserved as supplied; these are not labeled as Fama–French SMB/HML replications. Select the factors and weighting under `[factors]` before validation.

Attribution converts net TWD portfolio returns to unhedged USD: `(1 + return_twd) * FX_previous / FX_current - 1`, where FX is [FRED DEXTAUS](https://fred.stlouisfed.org/series/DEXTAUS), in TWD per USD. The first return uses the preceding exchange session's fixing. Only past FX observations are carried forward, for at most `fx_max_age_days` (default 7); factor returns are never filled.

The dependent variable subtracts a USD risk-free proxy: [Kenneth French monthly RF](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), divided by 100 and 21. This approximates JKP's underlying T-bill rate. Portfolio metrics remain in TWD. Reports disclose the proxy, broader JKP universe, and differences between FRED's New York fixing and JKP vendor FX. OLS point estimates have no significance tests and never affect model selection. JKP data use the provider's noncommercial license.

## Limitations and assumptions

- **Historical data:** dated TWSE listing/delisting records and ISIN classifications reconstruct membership without requiring present-day survival. Retired companies absent from ISIN use official company records and TWSE security-code rules; their classification starts at the first observed quote when the original listing date is unknown. Unverified candidates are excluded. Complete issuer lineage, historical type changes, code reuse, revisions, and delisting proceeds remain unaudited. An independent monthly calendar detects missing entire market sessions. See the [source details and audited-import schemas](docs/research-specification.md#canonical-inputs-and-public-source-limitations).
- **Corporate actions:** the parser uses explicit Chinese historical cash-dividend amounts and individual stock/rights terms; reference-price differences are never substituted for distributions. Payment dates, capital reductions, splits, and terminal settlements need additional evidence. The engine assumes **symmetric full subscription**: longs pay for all subscribed shares; shorts receive matching cash and owe the additional shares. Optional/partial exercise, rights sales, and funding constraints are not modeled.
- **Terminal events:** events are assumed **always recorded and known before that session's opening**, with proceeds settled and positions removed before opening NAV and orders. Date-only inputs cannot verify announcement timing; delayed settlement is not modeled.
- **Short selling:** borrowing is simulated with configurable charges. Historical availability, recalls, collateral, and actual short-side rights arrangements remain unresolved.
- **Execution:** fractional holdings and daily-price opening-fill approximations do not establish auction liquidity, capacity, or executable historical fills. Fees and slippage are assumptions; cash and short proceeds earn zero interest. Results remain exploratory/provisional.

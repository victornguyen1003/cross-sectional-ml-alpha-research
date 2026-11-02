# Cross-sectional ML Alpha Research

A reproducible, exploratory pipeline for Taiwan equity return ranking. Stage one implements data collection, price/volume features, forward labels, modeling, long–short construction, daily accounting, and evaluation. AI-agent integration is reserved for stage two.

The research contract is in [AGENTS.md](AGENTS.md), the implementation requirements are in [ROADMAP.md](ROADMAP.md), and detailed data and execution conventions are in [docs/research-specification.md](docs/research-specification.md).

## Setup

Use Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run --locked pytest
```

The lockfile pins the complete environment. The implementation uses functions organized into the eleven modules specified by the roadmap under `src/alpha_research/`.

## Reproduce the pipeline offline

Generate a deterministic synthetic dataset, including a dividend, a split, a suspension, and a terminal-value event. This is a software demonstration, not Taiwan market evidence. Its calendar is a weekday calendar and its research dates intentionally differ from the historical configuration.

```bash
uv run --locked python scripts/generate_demo.py --output artifacts/demo
uv run --locked alpha-research fetch --config artifacts/demo/demo.toml --offline
uv run --locked alpha-research validate-data --config artifacts/demo/demo.toml
uv run --locked alpha-research build-dataset --config artifacts/demo/demo.toml
uv run --locked alpha-research run --config artifacts/demo/demo.toml --name validation
uv run --locked alpha-research run --config artifacts/demo/demo.toml --final-test --selection artifacts/demo/runs/validation/selection.json --name final-test
uv run --locked alpha-research report --config artifacts/demo/demo.toml --run artifacts/demo/runs/final-test
```

Choose a new demo output directory for another generation. Existing raw responses and named runs are protected against overwriting.

Each run saves input checksums, resolved configuration, code fingerprints and Git state, dependency versions, predictions, model artifacts, eligibility decisions, target weights, holdings, trades, daily returns, cost breakdowns, benchmark returns, predictive diagnostics, sensitivity results, a Markdown report, and a performance chart. Run-local artifacts live below the configured `runs` directory.

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

Compare momentum, OLS, Lasso, Ridge, ElasticNet, and LightGBM using seed 103. Hyperparameters are tuned only inside the training period using expanding chronological date folds and label-end purging. All stocks on a signal date stay in the same fold. `model.search` supports exhaustive grid search or a seeded randomized LightGBM grid subset.

The training-only best settings for each model are compared on validation mean Spearman rank correlation. Ties favor momentum, OLS, Ridge, Lasso, ElasticNet, then LightGBM. Validation runs use a fixed training-period fit. The final run retrains the chosen specification on matured training and validation labels, then keeps that fit fixed throughout the test period.

A successful validation run produces `selection.json`. Final testing requires both `--final-test` and `--selection`. Configuration, input, or research-code changes invalidate the selection. This guard prevents accidental reuse after changes; it cannot prevent a researcher from repeatedly viewing the final test or manually editing selection files. Keep a research log and treat a viewed final test as spent for future specification selection.

Model files use Python pickle and should only be loaded from trusted runs.

## Interpretation

All reports are labeled exploratory/provisional. Missing historical actions, security identity, terminal values, revisions, actual borrow supply, auction liquidity, and historical fee schedules can materially affect results. The simulator uses fractional shares, synthetic borrowing, zero cash interest, and daily-price fill approximations. See the saved coverage report and research specification before interpreting performance. Positive returns are not a software acceptance criterion.

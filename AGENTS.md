# Project: AI Agents in ML-Based Alpha Research

## 1. Research Objective

This repository builds an alpha research pipeline from data collection to backtesting evaluation with LLM-based AI agent integration. The research focuses on ML modeling for cross-sectional equity return prediction and ranking using historical Taiwan stock market data.

The project has two stages:

1. Build a reproducible alpha research pipeline.
2. Integrate AI agents in each module of the pipeline.

## 2. Research Pipeline

Free TWSE data → Price/volume features → 20-session return rankings → Baseline models (OLS, Lasso, Ridge, ElasticNet, LightGBM) → Long–short portfolio → Cost-aware backtest → Evaluation.

Success means reproducible, auditable results with explicit limitations. Positive investment performance is not an acceptance requirement.

## 3. Research Contract

- Universe: TWSE-listed common stocks, excluding ETFs, warrants, preferred shares, and depositary receipts. Reconstruct historical membership from historical records rather than today's listings.
- Eligibility: Require 252 trading sessions of history and select the 500 most liquid eligible stocks using trailing 60-session average traded value, calculated through the signal date.
- Period: Collect 2010–2025; use 2010 for warm-up, 2011–2019 for initial training, 2020–2023 for validation, and 2024–2025 for the final test.
- Timing: Calculate signals after session `t` closes; enter at the next session's open. Predict total returns from open `t+1` through open `t+21`.
- Schedule: Generate signals every 20 exchange sessions, anchored to the first trading session of 2011. Use the same schedule for training samples and portfolio rebalancing.
- Portfolio: Equally weight the highest-ranked decile at +50% of NAV and the lowest-ranked decile at −50%. Target 100% gross exposure and zero net exposure at rebalance; allow weights to drift between rebalances.
- Interpretation: Use fractional holdings and simulated borrowing. Results remain exploratory wherever historical data or execution feasibility is incomplete.

## 4. Research Integrity

- Prevent look-ahead bias and data leakage.
- Use chronological training, validation, and held-out testing with appropriate purging.
- Ensure consistent feature timestamps, prediction targets, and execution assumptions.
- Use consistent price adjustment and corporate-action conventions.
- Include transaction costs in backtesting.
- Make experiments reproducible with explicit configurations and random seeds.
- Keep final test results out of iterative experiment selection.
- Document limitations of the available historical data rather than silently assuming they are resolved.

## 5. Engineering Principles

- Prioritize simplicity, readability, correctness, and reproducibility.
- Prefer simple functions and established Python libraries.
- Avoid unnecessary classes, abstractions, registries, factories, and frameworks.
- Keep dependencies minimal.
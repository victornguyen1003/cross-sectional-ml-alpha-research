"""Predictive diagnostics, ledger summaries, reproducible sensitivity replays."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .backtest import simulate
from .factors import attribution_summary
from .models import mean_rank_ic
from .portfolio import target_weights
from .provenance import write_json


def predictive_metrics(predictions):
    usable = predictions.dropna(subset=["realized_return", "prediction"]).copy()
    ic_rows, deciles = [], []
    for date, group in usable.groupby("date"):
        ic = mean_rank_ic(group)
        ic_rows.append({"date": date, "rank_ic": ic, "stocks": len(group)})
        group = group.copy()
        # Stable ordinal buckets are deterministic even if model scores are tied.
        group = group.sort_values(["prediction", "security_id"])
        group["decile"] = np.minimum(np.arange(len(group)) * 10 // len(group) + 1, 10)
        deciles.extend({"date": date, "decile": int(bucket), "return": values.realized_return.mean()}
                       for bucket, values in group.groupby("decile"))
    return {"mean_rank_ic": mean_rank_ic(usable), "prediction_rows": len(predictions),
            "labeled_rows": len(usable), "missing_label_rows": int(predictions.realized_return.isna().sum())}, pd.DataFrame(ic_rows), pd.DataFrame(deciles)


def performance(daily, initial_nav):
    returns = daily["return"]
    years = len(daily) / 252
    volatility = returns.std(ddof=1) * np.sqrt(252)
    wealth = pd.concat([pd.Series([initial_nav]), daily.nav], ignore_index=True)
    drawdown = wealth / wealth.cummax() - 1
    return {
        "total_return": float(daily.nav.iloc[-1] / initial_nav - 1),
        "gross_total_return": float(daily.gross_nav.iloc[-1] / initial_nav - 1),
        "annualized_return": float((daily.nav.iloc[-1] / initial_nav) ** (1 / years) - 1),
        "annualized_volatility": float(volatility),
        "sharpe": float(returns.mean() * 252 / volatility) if volatility > 0 else None,
        "max_drawdown": float(drawdown.min()),
        "total_turnover": float(daily.turnover.sum()),
        "mean_gross_exposure": float(daily.gross_exposure.mean()),
        "mean_net_exposure": float(daily.net_exposure.mean()),
        "stale_position_days": int(daily.stale_positions.sum()),
        "costs": {column: float(daily[column].sum()) for column in ["commission", "sell_tax", "slippage", "borrow"]},
        "long_pnl": float(daily.long_pnl.sum()), "short_pnl": float(daily.short_pnl.sum()),
    }


def sensitivity(prices, actions, calendar, targets, costs, settings):
    rows = []
    for slip in settings["slippage_bps"]:
        for borrow in settings["borrow_rates"]:
            assumptions = {**costs, "slippage_bps": slip, "borrow_rate": borrow}
            ledger = simulate(prices, actions, calendar, targets, assumptions)
            rows.append({"slippage_bps": slip, "borrow_rate": borrow,
                         **performance(ledger["daily"], assumptions["initial_nav"])})
    return rows


def report(run_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_path)
    attribution = attribution_summary(root)
    metrics = json.loads((root / "metrics.json").read_text())
    coverage = json.loads((root / "coverage.json").read_text())
    daily = pd.read_parquet(root / "daily.parquet")
    created_at = datetime.now(timezone.utc)
    suffix = (f"-{created_at:%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:8]}"
              if (root / "report.md").exists() or (root / "performance.png").exists() else "")
    report_path, image_path = root / f"report{suffix}.md", root / f"performance{suffix}.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    initial = json.loads((root / "manifest.json").read_text())["config"]["costs"]["initial_nav"]
    axes[0].plot(daily.date, daily.nav / initial, label="Net")
    axes[0].plot(daily.date, daily.gross_nav / initial, label="Gross, same share path")
    for name in ["momentum", "equal_weight"]:
        path = root / f"benchmark_{name}_daily.parquet"
        if path.exists():
            benchmark = pd.read_parquet(path)
            axes[0].plot(benchmark.date, benchmark.nav / initial, label=name)
    axes[0].legend()
    axes[0].set_ylabel("Wealth / initial NAV")
    wealth = pd.Series([initial, *daily.nav])
    axes[1].fill_between(daily.date, (wealth / wealth.cummax() - 1).iloc[1:].to_numpy(), 0)
    axes[1].set_ylabel("Drawdown")
    synthetic = any("SYNTHETIC" in source for source in coverage.get("sources", []))
    title = "SYNTHETIC SOFTWARE DEMO" if synthetic else "Exploratory research simulation"
    fig.suptitle(f"{title} — data and execution limitations apply")
    with image_path.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=150)
    plt.close(fig)
    lines = [f"# {title}", "", "Status: exploratory / provisional.", "",
             f"Generated at: {created_at.isoformat()}.", "",
             "Run configuration and input fingerprints: [manifest.json](manifest.json).", "",
             f"Data sources: {', '.join(coverage.get('sources', ['UNKNOWN']))}.", "",
             "Metrics and assumptions: `metrics.json`, `manifest.json`, and `coverage.json`.", "",
             f"Net total return: {metrics['portfolio']['total_return']:.2%}.",
             f"Mean rank IC: {metrics['predictive']['mean_rank_ic']}.", "",
             f"![Performance]({image_path.name})", "", "## Data limitations", ""]
    lines.extend(f"- {note}" for note in coverage["notes"])
    lines.extend(["", "## Common-factor return attribution", "",
                  f"Daily alpha: {attribution['alpha_daily']:.6g}; "
                  f"annualized arithmetic alpha: {attribution['alpha_annualized_arithmetic']:.6g}.",
                  f"R-squared: {attribution['r_squared']}; annualized residual volatility: "
                  f"{attribution['residual_volatility_annualized']:.6g}.",
                  f"Complete matched dates: {attribution['observations']} / {attribution['portfolio_dates']} "
                  f"({attribution['start']} through {attribution['end']}).", "",
                  f"Return series: {attribution['return_column']}; risk-free column: "
                  f"{attribution['risk_free_column']}; zero risk-free assumed: {attribution['zero_risk_free_assumed']}.",
                  "OLS point estimates only; this mandatory diagnostic does not enter model selection.", ""])
    lines.extend(f"- {factor}: {beta:.6g}" for factor, beta in attribution["betas"].items())
    if "source" in attribution:
        source = attribution["source"]
        lines.extend(["", f"Factor source: {source['url']} ({source['currency']}, {source['units']}); "
                      f"retrieved {source['retrieved_at']}; SHA-256: {source['sha256']}."])
        lines.extend(["", f"Risk-free convention: {source['risk_free_convention']}.",
                      f"FX convention: {source['fx_convention']}.",
                      "Attribution converts TWD portfolio NAV returns to unhedged USD; portfolio performance above remains in TWD.",
                      "JKP factors use a broader Taiwan universe and vendor FX; FRED fixing times and the USD risk-free proxy can affect residuals."])
        lines.append(f"Provider dates excluded for incomplete selected factors: {source['excluded_incomplete_factor_dates']}; "
                     "all portfolio sessions are required and retained.")
        lines.extend(f"- {name}: {item['url']}" for name, item in source["sources"].items())
        if "conversion" in attribution:
            lines.extend(["", f"Conversion: `{attribution['conversion']['formula']}`; initial FX baseline session: "
                          f"{attribution['conversion']['initial_fx_date']}."])
    lines.extend(["", "Details and alignment counts: `factor_regression/summary.json`.", ""])
    lines.extend(["", "- Gross wealth adds cumulative costs back to the same executed share path; it is not a separately rebalanced cost-free strategy.",
                  "- Holdings are marked at the period end without forced liquidation; closing costs are not charged on open positions.",
                  "- Decile returns are descriptive labeled-sample diagnostics, not executable portfolio returns.",
                  "- No statistical confidence or live tradability is implied by a positive Sharpe ratio.", ""])
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path


def save_evaluation(out, config, result):
    """Backtest selected scores, benchmarks and sensitivities; save diagnostics."""
    paths, r = config["paths"], config["research"]
    dataset = Path(paths["dataset"])
    selected, chosen_params = result["model"], result["params"]
    predictions = result["predictions"]
    period = result["period"]
    final_test = period == "test"
    validation_start, test_start = pd.Timestamp(r["validation_start"]), pd.Timestamp(r["test_start"])
    normalized = Path(paths["normalized"])
    prices = pd.read_parquet(normalized / "prices.parquet")
    actions = pd.read_parquet(normalized / "actions.parquet")
    calendar = pd.DatetimeIndex(pd.read_parquet(normalized / "calendar.parquet").date)
    period_start = test_start if final_test else validation_start
    calendar = calendar[(calendar >= period_start) & ((calendar <= pd.Timestamp(r["end"])) if final_test else (calendar < test_start))]
    targets = target_weights(predictions, r["decile"])
    if targets.empty:
        raise ValueError("No portfolio targets; require enough finite scores for two deciles")
    ledger = simulate(prices, actions, calendar, targets, config["costs"])
    predictive, rank_ic, deciles = predictive_metrics(predictions)
    metrics = {"period": period, "model": selected, "params": chosen_params,
               "predictive": predictive, "portfolio": performance(ledger["daily"], config["costs"]["initial_nav"]),
               "sensitivity": sensitivity(prices, actions, calendar, targets, config["costs"], config["evaluation"])}
    for key, frame in ledger.items():
        frame.to_parquet(out / f"{key}.parquet", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    targets.to_parquet(out / "targets.parquet", index=False)
    rank_ic.to_parquet(out / "rank_ic.parquet", index=False)
    deciles.to_parquet(out / "deciles.parquet", index=False)
    pd.read_parquet(dataset / "eligibility.parquet").query("date >= @period_start").loc[
        lambda f: f.date.isin(calendar)].to_parquet(out / "eligibility.parquet", index=False)
    if not final_test:
        for model_name, predicted in result["model_predictions"].items():
            predicted.to_parquet(out / f"predictions_{model_name}.parquet", index=False)
    for benchmark in ["momentum", "equal_weight"]:
        benchmark_predictions = predictions.copy()
        benchmark_predictions["prediction"] = predictions.momentum if benchmark == "momentum" else 1.0
        weights = target_weights(benchmark_predictions, r["decile"], long_only=benchmark == "equal_weight")
        benchmark_result = simulate(prices, actions, calendar, weights, config["costs"])
        benchmark_result["daily"].to_parquet(out / f"benchmark_{benchmark}_daily.parquet", index=False)
        metrics[f"benchmark_{benchmark}"] = performance(benchmark_result["daily"], config["costs"]["initial_nav"])
    coverage = json.loads((normalized / "coverage.json").read_text())
    coverage["missing_evaluation_labels"] = int(predictions.realized_return.isna().sum())
    coverage["notes"].extend([
        "Missing forward labels remain in predictions and trading, but cannot enter model fitting or labeled diagnostics.",
        "Incomplete-label cross-sections can bias training ranks and predictive diagnostics.",
        "Dividend targets use a reinvested total-return index; the ledger holds distributions as cash/receivables.",
        "Opening fills use supplied opening-time limits/blocks; missing auction liquidity and limit history remain unresolved.",
        "Targets use pre-fee NAV. Costs can cause small departures from target gross and net weights.",
        "Symmetric full subscription is assumed: longs pay for all subscribed shares; shorts receive matching cash and owe the extra shares.",
        "Terminal events are assumed recorded and settled before that session's opening; intraday/after-close timing is not modeled.",
    ])
    write_json(out / "metrics.json", metrics)
    write_json(out / "coverage.json", coverage)
    write_json(out / "cv_scores.json", result["cv_scores"])
    write_json(out / "validation_scores.json", result["validation_scores"])

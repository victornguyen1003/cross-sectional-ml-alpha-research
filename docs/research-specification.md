# Stage-One Research Specification

## Research timing and universe

The default universe is historical TWSE common stocks, with ETFs, warrants, preferred shares, and depositary receipts excluded. Eligibility requires 252 observed positive-volume closing sessions and a complete 60-session traded-value window. Select up to 500 securities by trailing mean traded value, breaking ties by identifier. The four-digit code fallback excludes prefixes `00` and `91` and is explicitly an unverified candidate heuristic. It does not establish historical identity, listing history, or security type.

Signals occur every 20 exchange sessions, anchored to the first session on or after the configured training start. Features use data through that session's close. Entry is next-session open; the label ends 20 sessions after entry. Missing open observations produce missing labels; they never remove a security from the earlier signal universe. All selected securities remain in predictions and portfolio construction even when their labels are unavailable.

Features use 5/20/60/120/252-session cumulative total returns, momentum from session `t-252` to `t-20`, 20/60-session annualized return volatility, log trailing 20-session mean traded value, and current volume divided by its 20-session mean. Rolling windows use exchange sessions. Missing execution prices remain missing; carried marks support valuation and adjusted return indices only.

The daily total-return factor is `(close * share_multiplier + cash_dividend - subscription_ratio * subscription_price) / previous_close`. The open index uses the same event terms against the previous close index. Dividends are reinvested in this index; the portfolio ledger instead holds dividend claims until payment. Documented terminal proceeds are held as cash through later label-end dates, including zero recovery; entry still requires an observed raw opening price. Adjustment events are applied on their effective date, and later events never alter earlier feature values. This convention assumes exercised subscription rights; it does not model saleable rights prices.

## Canonical inputs and public-source limitations

Raw JSON responses are stored as exact bytes with adjacent `.meta.json` files recording endpoint, parameters, source, retrieval time, and SHA-256. Metadata is the completion marker. Interrupted entries are rejected rather than silently repaired. Daily reports use `afterTrading/MI_INDEX` with `type=ALLBUT0999`. Ex-right reference reports use `exRight/TWT49U` in monthly date ranges. Report dates use Gregorian or ROC-year conversion explicitly. Volume is shares; value is TWD, without an implicit thousands multiplier.

The public reference-price report is not a complete corporate-action ledger. The parser only accepts explicit distribution terms; it never equates a reference-price difference with a cash dividend. Unsupported source schemas/actions appear in `coverage.json`. Canonical imports under the configured raw directory supplement these reports:

- `securities.csv`: `security_id,valid_from,valid_to,security_type`, with inclusive nonoverlapping intervals and `security_type=common` for eligible verified records. Preserve identifiers as strings. The master annotates observed history; it cannot manufacture missing historical prices.
- `calendar.csv`: one sorted, unique `date` per exchange session. The validator fails on missing entire supplied sessions. Without this file the calendar is inferred from observed reports and cannot detect missing entire market days.
- `actions.csv`: `date,security_id,cash_dividend,share_multiplier,subscription_ratio,subscription_price,payment_date,terminal_value`. Supply explicit defaults of 0 for cash/subscription amounts and 1 for the multiplier. `share_multiplier` includes stock dividends, splits, reductions, and subscribed shares. Imported terms override public parsed terms for the same date/security key. Multiple same-day events must be combined into one audited record.

`payment_date` and `terminal_value` may be blank. A missing payment date leaves the distribution as a receivable/payable with unchanged NAV, while cash timing remains unresolved. `terminal_value` is the documented cash amount per post-action share at the effective terminal date. For mergers involving replacement securities, import an audited equivalent settlement or leave the case unresolved; automatic stock-for-stock conversion is not implemented. Tax treatment and short manufactured-dividend terms are simplified.

Normalized Parquet tables store prices, action terms, and the calendar. Key uniqueness, nonnegative traded amounts, positive observed prices, OHLC consistency, action terms, dated security intervals, and supplied-calendar coverage are checked. Unquoted/suspended sessions remain explicit in the research panel. Missing action history is never interpreted as certified absence of corporate actions. Historical revisions and ticker reuse require additional source evidence; the raw archive records retrieval time, not original historical knowledge time.

## Training and selection

Train on cross-sectional percentile ranks of forward total returns. Missing outcomes cannot enter fitting or diagnostics; ranks are consequently based on the labeled subset, which can be biased when delisted/suspended outcomes are missing. Report these counts explicitly.

Training-only tuning uses expanding date folds. For a validation fold starting at date `v`, every admitted training label must end strictly before `v`. Initial training labels must end before the validation boundary. Validation labels crossing into the final-test interval are masked. Final training combines training and validation samples whose labels end before the test boundary. The final model remains fixed during testing.

OLS and the penalized linear models use training-fitted median imputation and standardization. LightGBM uses the same preprocessing, one CPU thread, deterministic column-wise training, fixed boosting iterations, and no random internal validation split. Seed 103 controls random choices. Parameter grids live in TOML. Within-training CV chooses parameters by mean date-level Spearman IC. The validation period chooses among those models and momentum using the same metric; the documented model-order tie break favors simpler models.

## Portfolio and accounting

Sort finite scores ascending, breaking ties by identifier. Short `floor(N * 0.1)` lowest-ranked stocks and buy the same number of highest-ranked stocks, equally weighted within each side at 50% of pre-fee NAV. Insufficient universes produce no targets. Cash remains unallocated when an opening fill fails; successful trades are not resized to replace missing fills. Fees can cause small departures from target gross exposure. Positions drift naturally between rebalances.

The ledger accrues borrow fees on previous-close short market value for actual calendar days/365. Actions occur before opening trades. Cash dividends create signed receivables, including manufactured payments on shorts; payment dates settle them to cash on the first processed session on or after the date. Stock actions change quantities; subscription terms change cash. Documented terminal proceeds settle positions at the effective date. Unknown terminal events retain holdings at stale marks and remain provisional.

Opening fills require a valid raw open; imported `opening_volume` blocks fills when explicitly nonpositive. Imported `buy_blocked`/`sell_blocked` boolean flags apply when present. Imported `limit_up`/`limit_down` prices conservatively block buys/sales at the adverse opening limit. Limits, opening volume and blocking flags must be known at the opening auction. The day's later high, low, close and total volume never decide an opening fill. Without historical opening limits and auction data, fillability remains unresolved; a raw daily quote is insufficient to certify an executable trade.

Charge commissions and slippage on each absolute traded notional, and sell tax on all sales, including short openings. Covers pay buy-side costs. Reversals pay costs on the full quantity change. Defaults are 14.25 bps commission, 30 bps sale tax, 10 bps slippage, and 3% annual borrow. Sensitivity replays use slippage 0/10/25 bps crossed with borrow 0%/3%/6%, rerunning the ledger because costs alter subsequent position sizing. These scenarios are diagnostics, not a basis for selecting specifications on the final test.

Assume zero interest on cash and short proceeds; no margin, settlement funding, position-size liquidity limit, board lots, borrow recalls, or actual locate checks are modeled. Fractional shares and gross exposure of 100% describe a research simulation, not broker feasibility. Nonpositive NAV stops a run as insolvent. The test begins in cash and finishes with positions marked rather than forcibly liquidated; unrealized terminal uncertainty and outstanding claims remain visible.

Gross wealth is net NAV plus cumulative costs for the same executed share path, not a separately rebalanced cost-free portfolio. Long/short P&L contributions exclude fees, taxes, slippage and borrow, which are reported separately. Turnover is absolute traded notional divided by opening pre-trade NAV, without a one-half multiplier. Annualization uses 252 sessions; drawdown includes initial NAV. Sharpe assumes a zero risk-free rate and does not supply a significance test.

## Reproducibility and acceptance

Run manifests include resolved configuration, seed, dependency versions, Git revision/dirty state, source fingerprint, and checksums of all raw, normalized, and dataset files. Separate validation/test directories contain predictions, selection, models, accounting ledgers, diagnostics, benchmarks, sensitivity results, and reports. Selection files bind the configuration, source code, and inputs. The final-test CLI requires an explicit frozen selection and rejects mismatches. Organizational discipline is still required to avoid repeated test-set selection.

Tests use offline cached fixtures, future-data perturbations, purged date-fold assertions, hand-calculated accounting examples, and repeated complete CLI runs. The synthetic generator includes distributions, suspension and terminal settlement and never claims real market evidence. Complete the pipeline and tests before a full historical download. Successful software execution does not certify the historical data or establish profitable alpha.

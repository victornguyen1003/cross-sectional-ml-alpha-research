"""Daily fractional-share ledger with explicit missing fills and stale marks.

Rights subscriptions are symmetric and full: longs pay all subscription cash and
receive all new shares; shorts receive matching cash and owe the new shares.
Terminal events are assumed recorded and settled before that session's opening.
"""

import numpy as np
import pandas as pd


def simulate(prices, actions, calendar, targets, costs):
    dates = pd.DatetimeIndex(calendar)
    if dates.empty:
        raise ValueError("Backtest calendar is empty")
    quotes = {date: frame.set_index("security_id") for date, frame in prices.groupby("date")}
    events = {date: frame for date, frame in actions.groupby("date")}
    orders = {date: frame.set_index("security_id").weight.to_dict()
              for date, frame in targets.dropna(subset=["entry_date"]).groupby("entry_date")}
    shares, marks, last_seen = {}, {}, {}
    cash = float(costs["initial_nav"])
    receivables, trades, holdings, daily, diagnostics = [], [], [], [], []
    cumulative_cost = 0.0
    previous_date = None
    previous_nav = cash
    previous_gross_nav = cash
    terminal_ids = set()
    for date in dates:
        frame = quotes.get(date, pd.DataFrame())
        old_shares, old_marks = shares.copy(), marks.copy()
        interval = (date - previous_date).days if previous_date is not None else 0
        borrow = sum(max(-q * marks.get(sid, 0.0), 0.0) for sid, q in shares.items())
        borrow *= costs["borrow_rate"] * interval / 365
        cash -= borrow
        cumulative_cost += borrow
        action_income = {}
        for event in events.get(date, pd.DataFrame()).itertuples():
            sid = event.security_id
            quantity = shares.get(sid, 0.0)
            dividend = quantity * event.cash_dividend
            # Full subscription for both signs of quantity; no choice to abstain
            # or sell rights. Negative quantities reverse the cash/share flows.
            subscription = quantity * event.subscription_ratio * event.subscription_price
            if dividend:
                receivables.append({"security_id": sid, "amount": dividend,
                                    "due": event.payment_date})
            cash -= subscription
            action_income[sid] = dividend - subscription
            multiplier = event.share_multiplier
            shares[sid] = quantity * multiplier
            # Adjust stale marks mechanically when an ex-date has no raw quote.
            if sid in marks:
                marks[sid] = (marks[sid] - event.cash_dividend +
                              event.subscription_ratio * event.subscription_price) / multiplier
            if pd.notna(event.terminal_value):
                # An input date denotes a pre-open event, never an intraday one.
                terminal_ids.add(sid)
                cash += shares.get(sid, 0.0) * event.terminal_value
                action_income[sid] = action_income.get(sid, 0.0) + shares.get(sid, 0.0) * event.terminal_value
                shares[sid] = 0.0
                marks[sid] = 0.0
        unpaid = []
        for payment in receivables:
            if pd.notna(payment["due"]) and payment["due"] <= date:
                cash += payment["amount"]
            else:
                unpaid.append(payment)
        receivables = unpaid
        accrued = sum(payment["amount"] for payment in receivables)
        open_marks = marks.copy()
        if not frame.empty:
            for sid, row in frame.iterrows():
                if sid not in terminal_ids and pd.notna(row.open):
                    open_marks[sid] = row.open
        action_shares = shares.copy()
        open_nav = cash + accrued + sum(q * open_marks.get(sid, 0.0) for sid, q in shares.items())
        if open_nav <= 0:
            raise ValueError(f"Non-positive NAV at {date}; strategy is insolvent")
        commission = sell_tax = slippage = traded = 0.0
        if date in orders:
            target = orders[date]
            for sid in sorted(set(shares) | set(target)):
                quantity = shares.get(sid, 0.0)
                weight = target.get(sid, 0.0)
                quote = frame.loc[sid] if sid in frame.index else None
                if sid in terminal_ids:
                    continue
                if (quote is None or pd.isna(quote.open) or
                        (pd.notna(quote.get("opening_volume", np.nan)) and quote.opening_volume <= 0)):
                    if quantity or weight:
                        diagnostics.append({"date": date, "security_id": sid, "reason": "missing_open_or_opening_volume"})
                    continue
                desired = weight * open_nav / quote.open
                change = desired - quantity
                if abs(change * quote.open) < 1e-8:
                    continue
                buy = change > 0
                upper, lower = quote.get("limit_up", np.nan), quote.get("limit_down", np.nan)
                # Limits/flags must be known by the opening auction. The day's
                # later high, low and close never decide whether an opening fills.
                adverse_lock = ((buy and pd.notna(upper) and quote.open >= upper) or
                                (not buy and pd.notna(lower) and quote.open <= lower))
                blocked = bool(quote.get("buy_blocked" if buy else "sell_blocked", False))
                if blocked or adverse_lock:
                    diagnostics.append({"date": date, "security_id": sid, "reason": "adverse_limit_or_block"})
                    continue
                notional = abs(change) * quote.open
                fee = notional * costs["commission_bps"] / 10000
                tax = notional * costs["sell_tax_bps"] / 10000 if not buy else 0.0
                slip = notional * costs["slippage_bps"] / 10000
                cash -= change * quote.open + fee + tax + slip
                shares[sid] = desired
                marks[sid] = quote.open
                commission += fee
                sell_tax += tax
                slippage += slip
                traded += notional
                trade = {"date": date, "security_id": sid, "quantity": change,
                         "price": quote.open, "notional": notional, "commission": fee,
                         "sell_tax": tax, "slippage": slip}
                trades.append(trade)
        cumulative_cost += commission + sell_tax + slippage
        if not frame.empty:
            for sid, row in frame.iterrows():
                if sid not in terminal_ids and pd.notna(row.close):
                    marks[sid] = row.close
                    last_seen[sid] = date
        nav = cash + accrued + sum(q * marks.get(sid, 0.0) for sid, q in shares.items())
        if nav <= 0:
            raise ValueError(f"Non-positive closing NAV at {date}; strategy is insolvent")
        long_pnl = short_pnl = 0.0
        for sid, old_q in old_shares.items():
            # Overnight/action P&L belongs to the previous side; intraday P&L
            # belongs to the post-trade side, including a reversal at the open.
            pnl = action_shares.get(sid, 0.0) * open_marks.get(sid, 0.0) - old_q * old_marks.get(sid, 0.0)
            pnl += action_income.get(sid, 0.0)
            if old_q > 0:
                long_pnl += pnl
            elif old_q < 0:
                short_pnl += pnl
        for sid, quantity in shares.items():
            pnl = quantity * (marks.get(sid, 0.0) - open_marks.get(sid, marks.get(sid, 0.0)))
            if quantity > 0:
                long_pnl += pnl
            elif quantity < 0:
                short_pnl += pnl
        gross_value = sum(abs(q * marks.get(sid, 0.0)) for sid, q in shares.items())
        net_value = sum(q * marks.get(sid, 0.0) for sid, q in shares.items())
        stale = 0
        for sid, quantity in sorted(shares.items()):
            if abs(quantity) < 1e-12:
                continue
            is_stale = last_seen.get(sid) != date
            stale += int(is_stale)
            holdings.append({"date": date, "security_id": sid, "quantity": quantity,
                             "mark": marks.get(sid, np.nan), "weight": quantity * marks.get(sid, 0.0) / nav,
                             "stale": is_stale})
        gross_nav = nav + cumulative_cost
        daily.append({"date": date, "nav": nav, "gross_nav": gross_nav,
                      "return": nav / previous_nav - 1,
                      "gross_return": gross_nav / previous_gross_nav - 1,
                      "cash": cash, "receivables": accrued, "commission": commission,
                      "sell_tax": sell_tax, "slippage": slippage, "borrow": borrow,
                      "turnover": traded / open_nav, "gross_exposure": gross_value / nav,
                      "net_exposure": net_value / nav, "stale_positions": stale,
                      "long_pnl": long_pnl, "short_pnl": short_pnl})
        previous_date, previous_nav, previous_gross_nav = date, nav, gross_nav
    return {"daily": pd.DataFrame(daily),
            "holdings": pd.DataFrame(holdings, columns=["date", "security_id", "quantity", "mark", "weight", "stale"]),
            "trades": pd.DataFrame(trades, columns=["date", "security_id", "quantity", "price", "notional", "commission", "sell_tax", "slippage"]),
            "execution_diagnostics": pd.DataFrame(diagnostics, columns=["date", "security_id", "reason"])}

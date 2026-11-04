"""Eligibility uses only observations through the signal close."""

import re
from html.parser import HTMLParser

import pandas as pd


class ISINTable(HTMLParser):
    """Read the official ten-column classification table without HTML dependencies."""

    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        if tag == "tr" and len(self.row) == 10 and self.row[0].isdigit():
            self.rows.append(self.row)


def security_history(isin_html, listing_payloads, delisting_payload, end, observations=None):
    """Reconstruct dated membership; never require a stock to survive to today."""
    def date(value):
        year, month, day = map(int, re.findall(r"\d+", value))
        return pd.Timestamp(year=year + 1911 if year < 1911 else year, month=month, day=day)
    ends = {}
    fields = delisting_payload.get("fields", [])
    if delisting_payload.get("data") and not {"Delisting Date", "Code"}.issubset(fields):
        raise ValueError("Unrecognized historical delisting fields")
    for row in delisting_payload.get("data", []):
        ends.setdefault(str(row[fields.index("Code")]).strip(), []).append(date(row[fields.index("Delisting Date")]))
    records = []
    observed = (observations.groupby("security_id").date.agg(["min", "max"])
                if observations is not None else pd.DataFrame())
    # An official delisted-company record plus TWSE's published ordinary-share
    # code rules identifies historical shares missing from today's ISIN table.
    # This bounds classification by observed quotes, not an invented IPO date.
    for sid, dates in ends.items():
        if not observed.empty and sid in observed.index:
            kind = ("depositary_receipt" if sid.startswith(("91", "92")) else
                    "common" if re.fullmatch(r"[1-9]\d{3}", sid) else "other")
            for stop in dates:
                if observed.loc[sid, "min"] < stop:
                    records.append({"security_id": sid, "valid_from": observed.loc[sid, "min"],
                                    "security_type": kind,
                                    "source": "TWSE delisted-company record and published code rules; classification starts at first observed quote, IPO date unknown"})
                    break
    # This historical company-listing report covers ordinary company shares,
    # including foreign primary listings, rather than ETF/TDR product listings.
    for payload in listing_payloads:
        fields = payload.get("fields", [])
        if not payload.get("data"):
            continue
        if not {"公司代號", "股票上市買賣日期"}.issubset(fields):
            raise ValueError("Unrecognized historical company-listing fields")
        for row in payload["data"]:
            start = str(row[fields.index("股票上市買賣日期")]).strip()
            if start:
                records.append({"security_id": str(row[fields.index("公司代號")]).strip(),
                                "valid_from": date(start), "security_type": "common", "source": "TWSE dated company listing"})
    if isin_html is not None:
        table = ISINTable()
        table.feed(isin_html)
        if not table.rows:
            raise ValueError("Unrecognized ISIN classification table")
        for row in table.rows:
            _, isin, sid, name, market, kind, industry, start, cfi, _ = row
            if not start or (market not in {"上市", "上市臺灣創新板"} and sid not in ends):
                continue
            security_type = "common" if cfi.startswith("ES") else "other"
            if "特別股" in kind or cfi.startswith("EP"):
                security_type = "preferred"
            if "TDR" in kind:
                security_type = "depositary_receipt"
            records.append({"security_id": sid, "valid_from": date(start), "security_type": security_type,
                            "source": f"TWSE ISIN {isin} / {cfi}; historical type continuity not independently certified"})
            if market in {"上市", "上市臺灣創新板"} and sid in ends and not observed.empty and sid in observed.index:
                # A later observed/current listing must not be truncated by an
                # older same-code delisting (e.g. 2301's 2002 reorganization).
                earlier = [stop for stop in ends[sid] if stop < observed.loc[sid, "max"]]
                if earlier:
                    records[-1]["valid_from"] = max(date(start), max(earlier) + pd.Timedelta(days=1))
    columns = ["security_id", "valid_from", "valid_to", "security_type", "source"]
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records).drop_duplicates(["security_id", "valid_from"], keep="last")
    frame = frame.sort_values(["security_id", "valid_from"])
    frame["valid_to"] = pd.Timestamp(end)
    for sid, group in frame.groupby("security_id"):
        for position, (index, row) in enumerate(group.iterrows()):
            boundaries = [d - pd.Timedelta(days=1) for d in ends.get(sid, []) if d >= row.valid_from]
            if position + 1 < len(group):
                boundaries.append(group.iloc[position + 1].valid_from - pd.Timedelta(days=1))
            frame.loc[index, "valid_to"] = min([pd.Timestamp(end), *boundaries])
    return frame[frame.valid_from <= frame.valid_to][columns].reset_index(drop=True)


def select_universe(panel, settings):
    frame = panel.copy()
    observed = frame.close.notna() & frame.volume.gt(0)
    frame["history_sessions"] = observed.groupby(frame.security_id).cumsum()
    frame["liquidity"] = frame.groupby("security_id").value.transform(
        lambda s: s.rolling(settings["liquidity_window"], min_periods=settings["liquidity_window"]).mean())
    frame["eligible"] = (frame.security_type.eq("common") & frame.identity_verified.fillna(False).astype(bool) &
                         frame.history_sessions.ge(settings["min_history"]) & observed &
                         frame.liquidity.gt(0))
    ranked = frame.loc[frame.eligible].sort_values(
        ["date", "liquidity", "security_id"], ascending=[True, False, True])
    selected = ranked.groupby("date").head(settings["universe_size"]).index
    frame["selected"] = frame.index.isin(selected)
    return frame

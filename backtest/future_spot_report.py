#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os, sys
import pandas as pd
import numpy as np
import json

# --------- CONFIG ---------
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()) / "alt_coins.json"
_DEFAULT_SYMBOLS = ["ZRXUSDT"]  # fallback if file missing/invalid
try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Clean, uppercase, dedupe while preserving order
    _symbols = [s.strip().upper() for s in data.get("alt_coins", []) if isinstance(s, str) and s.strip()]
    SYMBOLS = list(dict.fromkeys(_symbols)) or _DEFAULT_SYMBOLS
except Exception as e:
    print(f"[WARN] Could not load {JSON_PATH.name}: {e}. Using fallback list.")
    SYMBOLS = _DEFAULT_SYMBOLS

INTERVAL = "1m"
LOOKBACK_MINUTES = 60 * 24 * 30  # 30 days
RUN_DOWNLOAD = True              # set False to use existing CSVs

# Fees (taker-only; override with your actual VIP/discount rates)
SPOT_TAKER = 0.0010   # 0.10%
FUT_TAKER  = 0.0004   # 0.04%

# Threshold choice
USE_FUNDING_THRESHOLD = False    # True => use fee - funding (crit_net). False => fee-only.

# Detection side: only trade when perp > spot (spread positive)
REQUIRE_PERP_GT_SPOT = True

# Reports output
REPORTS_DIR = OUT_DIR / "arb_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
WRITE_SYMBOL_REPORTS = False  # if False, only the combined summary CSV is written

CLEANUP_RAW_AFTER_REPORT = True
# --------------------------

# Ensure we can import your downloader modules if present
sys.path.insert(0, str(Path("..").resolve()))
try:
    from binance_perp_ingest_full import fetch_perpetual_klines
    from binance_spot_ingest_full import fetch_spot_klines
except Exception as e:
    fetch_perpetual_klines = None
    fetch_spot_klines = None
    print("[WARN] Could not import fetchers; set RUN_DOWNLOAD=False to skip downloads.", e)

# ============ LOADERS (with correct timestamp handling) ============
def load_perp_csv(symbol: str, interval: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / f"{symbol}_{interval}_PERP.csv"
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trade_count","taker_base_volume","taker_quote_volume","ignore"]
    df = pd.read_csv(path, names=cols, header=0)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("open_time").sort_index()

def load_spot_csv(symbol: str, interval: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / f"{symbol}_{interval}_SPOT.csv"
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trade_count","taker_base_volume","taker_quote_volume","ignore"]
    df = pd.read_csv(path, names=cols, header=0)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("open_time").sort_index()

def load_funding_csv(symbol: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / f"{symbol}_funding.csv"
    cols = ["funding_time","funding_rate"]
    df = pd.read_csv(path, names=cols, header=0)
    df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df.set_index("funding_time").sort_index()

def cleanup_symbol_data(symbol: str, interval: str, out_dir: Path) -> None:
    """Delete raw CSVs for a symbol; keep reports."""
    files = [
        out_dir / f"{symbol}_{interval}_PERP.csv",
        out_dir / f"{symbol}_{interval}_SPOT.csv",
        out_dir / f"{symbol}_funding.csv",
    ]
    for p in files:
        try:
            p.unlink(missing_ok=True)
            print(f"[{symbol}] Deleted {p.name}")
        except Exception as e:
            print(f"[{symbol}] Could not delete {p.name}: {e}")

# ============ CORE COMPUTATION ============

def build_px(spot_df: pd.DataFrame, perp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return px with columns spot, perp; plus spread and pref series."""
    common_index = spot_df.index.intersection(perp_df.index)
    spot = spot_df.loc[common_index, ["close"]].rename(columns={"close": "spot"})
    perp = perp_df.loc[common_index, ["close"]].rename(columns={"close": "perp"})
    px = spot.join(perp, how="inner")
    spread = (px["perp"] - px["spot"]).rename("spread_usdt")
    pref   = ((px["perp"] + px["spot"]) / 2.0).rename("pref_usdt")
    return px, spread, pref

def compute_thresholds(pref: pd.Series, funding_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (crit_fee_only, crit_net) aligned to the px/pref index."""
    crit_fee_only = (pref * 2.0 * (SPOT_TAKER + FUT_TAKER)).rename("crit_fee_usdt")
    # funding ffilled onto pref index
    funding_ff = funding_df["funding_rate"].reindex(pref.index, method="ffill").fillna(0.0)
    crit_net = (crit_fee_only - (pref * funding_ff)).rename("crit_net_usdt")
    return crit_fee_only, crit_net

def find_tradable_events(spread: pd.Series, threshold: pd.Series) -> pd.DataFrame:
    """Return contiguous intervals where spread > threshold (and >0 if configured)."""
    thr = threshold.reindex(spread.index, method="ffill")
    tradable = spread > thr
    if REQUIRE_PERP_GT_SPOT:
        tradable &= (spread > 0)

    starts = tradable & ~tradable.shift(1, fill_value=False)
    ends   = tradable & ~tradable.shift(-1, fill_value=False)

    start_idx = spread.index[starts]
    end_idx   = spread.index[ends]
    events = pd.DataFrame({"start": start_idx, "end": end_idx})
    return events

def next_bar_after(index: pd.DatetimeIndex, t0) -> pd.Timestamp:
    """First bar strictly after t0; if not found, return t0 (last bar edge-case)."""
    pos = index.get_indexer([t0])[0]
    if pos == -1:
        pos = index.get_indexer([t0], method="nearest")[0]
    pos2 = min(pos + 1, len(index) - 1)
    return index[pos2]

def analyze_events(symbol: str,
                   px: pd.DataFrame,
                   spread: pd.Series,
                   pref: pd.Series,
                   crit_fee_only: pd.Series,
                   crit_net: pd.Series,
                   df_funding: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-event PnL/return/duration & symbol summary, save CSVs."""
    threshold = crit_net if USE_FUNDING_THRESHOLD else crit_fee_only
    events = find_tradable_events(spread, threshold)

    if events.empty:
        summary = pd.DataFrame([{
            "symbol": symbol,
            "n_events": 0,
            "avg_duration_min": 0.0,
            "median_duration_min": 0.0,
            "avg_half_life_min": np.nan,
            "total_pnl_usdt_per_coin": 0.0,
            "avg_pnl_usdt_per_coin": 0.0,
            "avg_return_pct_on_2legs": 0.0,
            "win_rate_pct": 0.0,
            # NEW: total PnL % over summed two-leg notional
            "total_pnl_pct_on_2legs": np.nan
        }])
        if WRITE_SYMBOL_REPORTS:
            (REPORTS_DIR / f"{symbol}_events.csv").write_text("")
            sum_path = REPORTS_DIR / f"{symbol}_summary.csv"
            summary.to_csv(sum_path, index=False)
            print(f"[{symbol}] No tradable events. Summary -> {sum_path}")
        return summary, events

    funding_series = df_funding["funding_rate"].sort_index()

    rows = []
    for _, ev in events.iterrows():
        t0 = ev["start"]
        t_last = ev["end"]
        t1 = next_bar_after(px.index, t_last)

        spot_entry = px.at[t0, "spot"]
        perp_entry = px.at[t0, "perp"]
        spot_exit  = px.at[t1, "spot"]
        perp_exit  = px.at[t1, "perp"]
        pref_entry = pref.at[t0]
        pref_exit  = pref.at[t1]

        fee_entry = pref_entry * (SPOT_TAKER + FUT_TAKER)
        fee_exit  = pref_exit  * (SPOT_TAKER + FUT_TAKER)
        fees_total = fee_entry + fee_exit

        f_slice = funding_series.loc[(funding_series.index > t0) & (funding_series.index <= t1)]
        if len(f_slice) > 0:
            pref_at_f = pref.reindex(f_slice.index, method="ffill")
            funding_usdt = float((pref_at_f * f_slice).sum())
        else:
            funding_usdt = 0.0

        pnl_perp = (perp_entry - perp_exit)     # short perp
        pnl_spot = (spot_exit - spot_entry)     # long spot
        pnl_gross = pnl_perp + pnl_spot + funding_usdt
        pnl_net = pnl_gross - fees_total

        # per-event two-leg notional at entry (for weighting)
        entry_notional_2legs = float(2.0 * pref_entry)          # NEW

        denom = entry_notional_2legs if entry_notional_2legs != 0 else np.nan
        ret_pct = float(pnl_net / denom * 100.0) if pd.notna(denom) else np.nan

        duration_min = (t1 - t0).total_seconds() / 60.0
        s0 = spread.at[t0]
        thr0 = threshold.at[t0]
        target = thr0 + 0.5 * (s0 - thr0)
        path = spread.loc[t0:t1]
        half_ts = path[path <= target].index.min()
        half_life_min = ((half_ts - t0).total_seconds() / 60.0) if pd.notna(half_ts) else np.nan

        rows.append({
            "symbol": symbol,
            "start": t0, "end": t1, "bars": int(len(px.loc[t0:t1])),
            "duration_min": duration_min,
            "half_life_min": half_life_min,
            "entry_spot": spot_entry, "entry_perp": perp_entry,
            "exit_spot": spot_exit,   "exit_perp": perp_exit,
            "spread_entry": s0, "spread_exit": spread.at[t1],
            "fees_entry_usdt": fee_entry, "fees_exit_usdt": fee_exit,
            "funding_usdt": funding_usdt,
            "pnl_usdt": pnl_net,
            "return_pct_on_2legs": ret_pct,
            "entry_notional_2legs": entry_notional_2legs        # NEW
        })

    ev_df = pd.DataFrame(rows).set_index("start").sort_index()

    # NEW: total PnL % on summed two-leg notional
    total_pnl_usdt = float(ev_df["pnl_usdt"].sum())
    total_notional_2legs = float(ev_df["entry_notional_2legs"].sum())
    total_pnl_pct_on_2legs = (total_pnl_usdt / total_notional_2legs * 100.0) if total_notional_2legs > 0 else np.nan

    summary = pd.DataFrame([{
        "symbol": symbol,
        "n_events": int(len(ev_df)),
        "avg_duration_min": float(ev_df["duration_min"].mean()),
        "median_duration_min": float(ev_df["duration_min"].median()),
        "avg_half_life_min": float(ev_df["half_life_min"].mean(skipna=True)),
        "total_pnl_usdt_per_coin": total_pnl_usdt,
        "avg_pnl_usdt_per_coin": float(ev_df["pnl_usdt"].mean()),
        "avg_return_pct_on_2legs": float(ev_df["return_pct_on_2legs"].mean()),
        "win_rate_pct": float((ev_df["pnl_usdt"] > 0).mean() * 100.0),
        "total_pnl_pct_on_2legs": float(total_pnl_pct_on_2legs)           # NEW
    }])

    if WRITE_SYMBOL_REPORTS:
        ev_path = REPORTS_DIR / f"{symbol}_events.csv"
        sum_path = REPORTS_DIR / f"{symbol}_summary.csv"
        ev_df.to_csv(ev_path)
        summary.to_csv(sum_path, index=False)
        print(f"[{symbol}] Saved events -> {ev_path}")
        print(f"[{symbol}] Saved summary -> {sum_path}")

    return summary, ev_df

# ============ MAIN PIPELINE ============

def run_for_symbol(symbol: str):
    # Download fresh data (optional)
    if RUN_DOWNLOAD:
        if fetch_perpetual_klines is None or fetch_spot_klines is None:
            raise RuntimeError("Downloader modules not available; set RUN_DOWNLOAD=False or fix imports.")
        print(f"[{symbol}] Downloading PERP+funding…")
        fetch_perpetual_klines(
            symbols=[symbol],
            out_dir=str(OUT_DIR),
            interval=INTERVAL,
            lookback_minutes=LOOKBACK_MINUTES,
            resume=True,
            pause=0.12,
            pause_between_symbols=0.25,
            retries=5,
            verbose=True,
            insecure_ssl=True,
            with_funding=True
        )
        print(f"[{symbol}] Downloading SPOT…")
        fetch_spot_klines(
            symbols=[symbol],
            out_dir=str(OUT_DIR),
            interval=INTERVAL,
            lookback_minutes=LOOKBACK_MINUTES,
            resume=True,
            pause=0.12,
            pause_between_symbols=0.25,
            retries=5,
            verbose=True,
            insecure_ssl=True
        )

    # Load CSVs
    df_perp = load_perp_csv(symbol, INTERVAL, OUT_DIR)
    df_spot = load_spot_csv(symbol, INTERVAL, OUT_DIR)
    df_funding = load_funding_csv(symbol, OUT_DIR)

    # Build aligned price frames
    px, spread, pref = build_px(df_spot, df_perp)

    # Thresholds
    crit_fee_only, crit_net = compute_thresholds(pref, df_funding)

    # Analyze events & save reports
    summary_df, events_df = analyze_events(
        symbol, px, spread, pref, crit_fee_only, crit_net, df_funding
    )

    if CLEANUP_RAW_AFTER_REPORT:
        cleanup_symbol_data(symbol, INTERVAL, OUT_DIR)

    return summary_df, events_df

def main():
    summaries = []
    for sym in SYMBOLS:
        try:
            summary_df, _ = run_for_symbol(sym)
            summaries.append(summary_df)
        except Exception as e:
            print(f"[{sym}] ERROR: {e}")

    if summaries:
        all_summary = pd.concat(summaries, ignore_index=True)
        path = REPORTS_DIR / "ALL_SYMBOLS_summary.csv"
        all_summary.to_csv(path, index=False)
        print(f"[ALL] Saved combined summary -> {path}")

if __name__ == "__main__":
    main()

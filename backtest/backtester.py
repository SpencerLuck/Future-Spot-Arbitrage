#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple per-symbol backtester for spot-perp arbitrage.

Assumptions:
- One symbol at a time (no portfolio overlap)
- Percent equity sizing
- Slippage + taker fees on entry/exit for both legs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import sys

import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
from backtester_config import BacktestConfig, load_symbols

CFG = BacktestConfig()

# -------------------------
# Optional ingest imports
# -------------------------
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
try:
    from binance_perp_ingest_full import fetch_perpetual_klines
    from binance_spot_ingest_full import fetch_spot_klines
except Exception as e:
    fetch_perpetual_klines = None
    fetch_spot_klines = None
    print("[WARN] Could not import fetchers; set run_download=False to use existing CSVs.", e)

# =========================
# Helpers
# =========================
def apply_slippage(price: float, side: str, slip: float) -> float:
    if side == "buy":
        return price * (1.0 + slip)
    if side == "sell":
        return price * (1.0 - slip)
    raise ValueError(f"Unknown side: {side}")

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
    if not path.exists():
        return pd.DataFrame(columns=cols).set_index("funding_time")
    df = pd.read_csv(path, names=cols, header=0)
    df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df.set_index("funding_time").sort_index()

def align_price_frames(df_spot: pd.DataFrame, df_perp: pd.DataFrame) -> pd.DataFrame:
    spot = df_spot["close"].rename("spot")
    perp = df_perp["close"].rename("perp")
    idx = spot.index.intersection(perp.index)
    spot = spot.reindex(idx)
    perp = perp.reindex(idx)
    pref = ((spot + perp) / 2.0).rename("pref")
    spread = (perp - spot).rename("spread")
    return pd.concat([spot, perp, pref, spread], axis=1).dropna()

def compute_thresholds(pref: pd.Series, funding: pd.DataFrame,
                       spot_taker: float, fut_taker: float,
                       mode: str = "net") -> pd.Series:
    fee_only = (pref * 2.0 * (spot_taker + fut_taker)).rename("crit_fee_usdt")
    if mode == "fee-only":
        return fee_only.reindex(pref.index, method="ffill")
    if funding is None or funding.empty:
        return fee_only.reindex(pref.index, method="ffill")
    funding_ff = funding["funding_rate"].reindex(pref.index, method="ffill").fillna(0.0)
    crit_net = (fee_only - (pref * funding_ff)).rename("crit_net_usdt")
    return crit_net

def find_tradable_events(spread: pd.Series, threshold: pd.Series,
                         require_perp_gt_spot: bool = True,
                         min_bars: int = 1) -> pd.DataFrame:
    thr = threshold.reindex(spread.index, method="ffill")
    tradable = spread > thr
    if require_perp_gt_spot:
        tradable &= spread > 0

    starts = tradable & ~tradable.shift(1, fill_value=False)
    ends = tradable & ~tradable.shift(-1, fill_value=False)

    start_idx = spread.index[starts]
    end_idx = spread.index[ends]
    events = pd.DataFrame({"start": start_idx, "end": end_idx})
    events = events[events["end"] >= events["start"]]
    if min_bars > 1 and not events.empty:
        spans = []
        for _, row in events.iterrows():
            bars = len(spread.loc[row["start"]:row["end"]])
            spans.append(bars)
        events = events.assign(bars=spans)
        events = events[events["bars"] >= min_bars].drop(columns=["bars"])
    return events.reset_index(drop=True)

def next_bar_after(index: pd.DatetimeIndex, t0) -> pd.Timestamp:
    loc = index.get_indexer([t0])
    if loc[0] == -1:
        pos = int(index.searchsorted(t0, side="left"))
    else:
        pos = int(loc[0])
    pos2 = min(pos + 1, len(index) - 1)
    return index[pos2]

def resolve_time_window(cfg: BacktestConfig) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    start_ts = None
    end_ts = None
    if cfg.start_ms is not None:
        start_ts = pd.to_datetime(cfg.start_ms, unit="ms")
    if cfg.end_ms is not None:
        end_ts = pd.to_datetime(cfg.end_ms, unit="ms")
    if cfg.start_date:
        start_ts = pd.to_datetime(cfg.start_date)
    if cfg.end_date:
        end_ts = pd.to_datetime(cfg.end_date)
    return start_ts, end_ts

def ts_to_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)

def have_symbol_files(symbol: str, cfg: BacktestConfig) -> bool:
    spot_path = cfg.spot_dir / f"{symbol}_{cfg.interval}_SPOT.csv"
    perp_path = cfg.perp_dir / f"{symbol}_{cfg.interval}_PERP.csv"
    return spot_path.exists() and perp_path.exists()

def cleanup_symbol_data(symbol: str, cfg: BacktestConfig) -> None:
    files = [
        cfg.perp_dir / f"{symbol}_{cfg.interval}_PERP.csv",
        cfg.perp_dir / f"{symbol}_funding.csv",
        cfg.spot_dir / f"{symbol}_{cfg.interval}_SPOT.csv",
    ]
    for path in files:
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[{symbol}] Could not delete {path.name}: {e}")

@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: pd.DataFrame

def backtest_symbol(symbol: str, cfg: BacktestConfig) -> BacktestResult:
    df_spot = load_spot_csv(symbol, cfg.interval, cfg.spot_dir)
    df_perp = load_perp_csv(symbol, cfg.interval, cfg.perp_dir)
    df_funding = load_funding_csv(symbol, cfg.perp_dir)

    px = align_price_frames(df_spot, df_perp)
    if px.empty:
        raise ValueError("No overlapping spot/perp data after alignment.")

    start_ts, end_ts = resolve_time_window(cfg)
    if start_ts is not None or end_ts is not None:
        px = px.loc[start_ts:end_ts]
        df_funding = df_funding.loc[start_ts:end_ts] if not df_funding.empty else df_funding
        if px.empty:
            raise ValueError("No data after applying time window.")

    threshold = compute_thresholds(px["pref"], df_funding, cfg.spot_taker, cfg.fut_taker, cfg.threshold_mode)
    events = find_tradable_events(px["spread"], threshold, cfg.require_perp_gt_spot, cfg.min_bars)

    slip = cfg.slippage_bps / 10_000.0
    equity = float(cfg.start_equity)
    trades = []
    equity_rows = [{"timestamp": px.index[0], "equity": equity, "pnl": 0.0}]

    for _, ev in events.iterrows():
        entry_time = next_bar_after(px.index, ev["start"])
        exit_time = next_bar_after(px.index, ev["end"])
        if entry_time >= exit_time:
            continue

        spot_entry_px = float(px.at[entry_time, "spot"])
        perp_entry_px = float(px.at[entry_time, "perp"])
        spot_exit_px = float(px.at[exit_time, "spot"])
        perp_exit_px = float(px.at[exit_time, "perp"])
        pref_entry = float(px.at[entry_time, "pref"])

        alloc_notional = equity * cfg.equity_pct
        qty = alloc_notional / (2.0 * pref_entry) if pref_entry > 0 else 0.0
        if qty <= 0:
            continue

        spot_entry = apply_slippage(spot_entry_px, "buy", slip)
        perp_entry = apply_slippage(perp_entry_px, "sell", slip)
        spot_exit = apply_slippage(spot_exit_px, "sell", slip)
        perp_exit = apply_slippage(perp_exit_px, "buy", slip)

        pnl_spot = qty * (spot_exit - spot_entry)
        pnl_perp = qty * (perp_entry - perp_exit)

        f_slice = df_funding.loc[(df_funding.index > entry_time) & (df_funding.index <= exit_time)]
        if len(f_slice) > 0:
            pref_at_f = px["pref"].reindex(f_slice.index, method="ffill")
            funding_usdt = float((pref_at_f * f_slice["funding_rate"]).sum()) * qty
        else:
            funding_usdt = 0.0

        fees = qty * (
            spot_entry * cfg.spot_taker +
            perp_entry * cfg.fut_taker +
            spot_exit * cfg.spot_taker +
            perp_exit * cfg.fut_taker
        )

        pnl_net = pnl_spot + pnl_perp + funding_usdt - fees
        equity_start = equity
        equity = equity + pnl_net
        ret_pct_alloc = (pnl_net / alloc_notional * 100.0) if alloc_notional > 0 else np.nan

        trades.append({
            "symbol": symbol,
            "signal_start": ev["start"],
            "signal_end": ev["end"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "qty": qty,
            "alloc_notional": alloc_notional,
            "spot_entry": spot_entry,
            "perp_entry": perp_entry,
            "spot_exit": spot_exit,
            "perp_exit": perp_exit,
            "pnl_spot": pnl_spot,
            "pnl_perp": pnl_perp,
            "funding_usdt": funding_usdt,
            "fees_usdt": fees,
            "pnl_net": pnl_net,
            "equity_start": equity_start,
            "equity_end": equity,
            "return_pct_on_alloc": ret_pct_alloc,
        })
        equity_rows.append({"timestamp": exit_time, "equity": equity, "pnl": pnl_net})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        equity_df["run_max"] = equity_df["equity"].cummax()
        equity_df["drawdown_pct"] = (equity_df["equity"] - equity_df["run_max"]) / equity_df["run_max"] * 100.0

    if trades_df.empty:
        summary = pd.DataFrame([{
            "symbol": symbol,
            "n_trades": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "start_equity": cfg.start_equity,
            "end_equity": cfg.start_equity,
            "total_return_pct": 0.0
        }])
    else:
        total_pnl = float(trades_df["pnl_net"].sum())
        avg_pnl = float(trades_df["pnl_net"].mean())
        win_rate = float((trades_df["pnl_net"] > 0).mean() * 100.0)
        end_equity = float(trades_df["equity_end"].iloc[-1])
        total_return = (end_equity - cfg.start_equity) / cfg.start_equity * 100.0
        max_dd = float(equity_df["drawdown_pct"].min()) if "drawdown_pct" in equity_df else 0.0
        summary = pd.DataFrame([{
            "symbol": symbol,
            "n_trades": int(len(trades_df)),
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "win_rate_pct": win_rate,
            "max_drawdown_pct": max_dd,
            "start_equity": cfg.start_equity,
            "end_equity": end_equity,
            "total_return_pct": total_return
        }])

    return BacktestResult(trades=trades_df, equity_curve=equity_df, summary=summary)

def main() -> int:
    symbols = load_symbols(CFG)
    if not symbols:
        print("[WARN] No symbols configured. Update backtester_config.py.")
        return 0

    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = None
    end_ts = None
    start_ms = CFG.start_ms
    end_ms = CFG.end_ms
    if CFG.run_download:
        if fetch_spot_klines is None or fetch_perpetual_klines is None:
            raise RuntimeError("Downloader modules not available; set RUN_DOWNLOAD=False or fix imports.")
        start_ts, end_ts = resolve_time_window(CFG)
        if start_ts is not None:
            start_ms = ts_to_ms(start_ts)
        if end_ts is not None:
            end_ms = ts_to_ms(end_ts)

    all_trades = []
    all_summaries = []
    for sym in symbols:
        try:
            if CFG.run_download:
                print(f"[DL] {sym}: SPOT...")
                fetch_spot_klines(
                    symbols=[sym],
                    out_dir=str(CFG.spot_dir),
                    interval=CFG.interval,
                    lookback_minutes=CFG.lookback_minutes,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    resume=True,
                    pause=0.12,
                    pause_between_symbols=0.25,
                    retries=5,
                    verbose=True,
                    insecure_ssl=True
                )
                print(f"[DL] {sym}: PERP+funding...")
                fetch_perpetual_klines(
                    symbols=[sym],
                    out_dir=str(CFG.perp_dir),
                    interval=CFG.interval,
                    lookback_minutes=CFG.lookback_minutes,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    resume=True,
                    pause=0.12,
                    pause_between_symbols=0.25,
                    retries=5,
                    verbose=True,
                    insecure_ssl=True,
                    with_funding=True
                )
            elif not have_symbol_files(sym, CFG):
                print(f"[{sym}] SKIP: missing spot/perp CSVs")
                continue

            result = backtest_symbol(sym, CFG)
            if not result.trades.empty:
                all_trades.append(result.trades)
            all_summaries.append(result.summary)
            print(f"[{sym}] trades: {len(result.trades)}")
        except Exception as e:
            print(f"[{sym}] ERROR: {e}")
        if CFG.cleanup_raw_after_backtest and CFG.run_download:
            cleanup_symbol_data(sym, CFG)

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        trades_df = trades_df.sort_values(["entry_time", "symbol"])
        trades_path = CFG.output_dir / CFG.output_trades_name
        trades_df.to_csv(trades_path, index=False)
        print(f"[ALL] trades -> {trades_path}")
    else:
        print("[ALL] No trades generated.")

    if all_summaries:
        summary_df = pd.concat(all_summaries, ignore_index=True)
        summary_path = CFG.output_dir / CFG.output_summary_name
        summary_df.to_csv(summary_path, index=False)
        print(f"[ALL] summary -> {summary_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

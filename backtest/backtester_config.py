#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central config for the per-symbol backtester.
Edit values here to control sizing, timeframe, symbols, and outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class BacktestConfig:
    # sizing + costs
    start_equity: float = 10_000.0
    equity_pct: float = 0.10
    slippage_bps: float = 1.0
    spot_taker: float = 0.0010
    fut_taker: float = 0.0004

    # signal logic
    threshold_mode: str = "net"  # "net" or "fee-only"
    require_perp_gt_spot: bool = True
    min_bars: int = 1

    # data window
    interval: str = "1m"
    run_download: bool = True
    lookback_minutes: int = 60 * 24 * 30
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    start_date: Optional[str] = None  # "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"
    end_date: Optional[str] = None

    # symbols
    symbols: Optional[List[str]] = None
    use_alt_coins: bool = True
    alt_coins_path: Path = Path(__file__).resolve().parent / "alt_coins.json"

    # IO
    perp_dir: Path = Path("arbitrage/data")
    spot_dir: Path = Path("arbitrage/data_spot")
    output_dir: Path = Path("data/arb_trades")
    output_trades_name: str = "ALL_TRADES.csv"
    output_summary_name: str = "ALL_SUMMARY.csv"
    cleanup_raw_after_backtest: bool = True


def load_symbols(cfg: BacktestConfig) -> List[str]:
    if cfg.symbols:
        return [s.strip().upper() for s in cfg.symbols if s.strip()]
    if not cfg.use_alt_coins:
        return []
    try:
        with open(cfg.alt_coins_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        syms = [s.strip().upper() for s in data.get("alt_coins", []) if isinstance(s, str) and s.strip()]
        return list(dict.fromkeys(syms))
    except Exception as e:
        print(f"[WARN] Could not load {cfg.alt_coins_path.name}: {e}")
        return []

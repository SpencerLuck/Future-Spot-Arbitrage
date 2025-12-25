#!/usr/bin/env python3
"""Minimal listener that prints best bid/ask snapshots and funding events."""

from __future__ import annotations

import json
import time
from typing import Dict, Iterable, List, Optional

from fetch_best_quote_snapshot import fetch_best_quote_snapshot
from fetch_latest_funding_event import fetch_latest_funding_event


def _iter_market_types(market_type: str) -> Iterable[str]:
    mt = (market_type or "").strip().lower()
    if mt == "futures":
        mt = "perp"
    if mt == "both":
        return ("spot", "perp")
    if mt in ("spot", "perp"):
        return (mt,)
    raise ValueError(f"Unsupported market_type: {market_type}")


def _print_json_line(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def listen(
    symbols: List[str],
    *,
    market_type: str = "both",
    quote_interval_ms: int = 500,
    quote_depth_levels: int = 5,
    user_agent: str = "arbitrage-listener/1.0",
    insecure_ssl: bool = True,
    retries: int = 3,
    pause: float = 0.5,
    with_funding: bool = False,
    funding_interval_ms: int = 60_000,
    include_index_price: bool = False
) -> None:
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")

    quote_interval_s = max(quote_interval_ms, 1) / 1000.0
    funding_interval_s = max(funding_interval_ms, 1) / 1000.0

    next_quote = time.time()
    next_funding = time.time()
    last_funding: Dict[str, Optional[int]] = {s: None for s in symbols}
    iter_id = 0

    request_opts = {
        "user_agent": user_agent,
        "insecure_ssl": insecure_ssl,
        "retries": retries,
        "pause": pause,
    }

    while True:
        now = time.time()

        if now >= next_quote:
            iter_id += 1
            for symbol in symbols:
                for mt in _iter_market_types(market_type):
                    snapshot = fetch_best_quote_snapshot(
                        symbol,
                        mt,
                        depth_levels=quote_depth_levels,
                        **request_opts,
                    )
                    snapshot["iter_id"] = iter_id
                    _print_json_line(snapshot)
            next_quote += quote_interval_s
            if next_quote < now:
                next_quote = now + quote_interval_s

        if with_funding and now >= next_funding:
            for symbol in symbols:
                event = fetch_latest_funding_event(
                    symbol,
                    include_index_price=include_index_price,
                    **request_opts,
                )
                if not event:
                    continue
                funding_time = event.get("funding_time")
                if funding_time and funding_time != last_funding.get(symbol):
                    _print_json_line(event)
                    last_funding[symbol] = funding_time
            next_funding += funding_interval_s
            if next_funding < now:
                next_funding = now + funding_interval_s

        sleep_until = min(next_quote, next_funding) if with_funding else next_quote
        time.sleep(max(0.0, sleep_until - time.time()))


def main() -> None:
    listen(["BTCUSDT"], quote_interval_ms=1000, with_funding=True)


if __name__ == "__main__":
    main()

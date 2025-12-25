#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, List

from binance_http import DEFAULT_INSECURE_SSL, http_get_with_retry

SPOT_BASE_DEFAULT = "https://api.binance.com"
FAPI_BASE_DEFAULT = "https://fapi.binance.com"


def _parse_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _parse_levels(levels: list, depth_levels: int) -> Tuple[List[float], List[float]]:
    prices: List[float] = []
    qtys: List[float] = []
    for level in levels[:depth_levels]:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price = _parse_float(level[0])
        qty = _parse_float(level[1])
        if price is None or qty is None:
            continue
        prices.append(price)
        qtys.append(qty)
    return prices, qtys


def _normalize_market_type(market_type: str) -> str:
    mt = (market_type or "").strip().lower()
    if mt == "futures":
        return "perp"
    if mt in ("spot", "perp"):
        return mt
    raise ValueError(f"Unsupported market_type: {market_type}")


def fetch_best_quote_snapshot(
    symbol: str,
    market_type: str,
    *,
    spot_base: str = SPOT_BASE_DEFAULT,
    perp_base: str = FAPI_BASE_DEFAULT,
    user_agent: str = "arbitrage-listener/1.0",
    insecure_ssl: bool = DEFAULT_INSECURE_SSL,
    depth_levels: int = 5,
    retries: int = 3,
    pause: float = 0.5,
) -> Dict[str, object]:
    symbol = symbol.strip().upper()
    mt = _normalize_market_type(market_type)
    path = "/api/v3/ticker/bookTicker" if mt == "spot" else "/fapi/v1/ticker/bookTicker"
    base = spot_base if mt == "spot" else perp_base

    data = http_get_with_retry(
        path,
        {"symbol": symbol},
        base=base,
        user_agent=user_agent,
        insecure_ssl=insecure_ssl,
        retries=retries,
        pause=pause,
    )

    exchange_ts = None
    if isinstance(data, dict) and "time" in data:
        try:
            exchange_ts = int(data["time"])
        except Exception:
            exchange_ts = None

    payload: Dict[str, object] = {
        "ts": int(time.time() * 1000),
        "exchange_ts": exchange_ts,
        "symbol": symbol,
        "market_type": mt,
        "bid_price": _parse_float(data.get("bidPrice")),
        "bid_qty": _parse_float(data.get("bidQty")),
        "ask_price": _parse_float(data.get("askPrice")),
        "ask_qty": _parse_float(data.get("askQty")),
    }

    if depth_levels and depth_levels > 0:
        depth_path = "/api/v3/depth" if mt == "spot" else "/fapi/v1/depth"
        depth = http_get_with_retry(
            depth_path,
            {"symbol": symbol, "limit": depth_levels},
            base=base,
            user_agent=user_agent,
            insecure_ssl=insecure_ssl,
            retries=retries,
            pause=pause,
        )
        if isinstance(depth, dict):
            bid_prices, bid_qtys = _parse_levels(depth.get("bids", []), depth_levels)
            ask_prices, ask_qtys = _parse_levels(depth.get("asks", []), depth_levels)
            payload.update(
                {
                    "bid_prices": bid_prices,
                    "bid_qtys": bid_qtys,
                    "ask_prices": ask_prices,
                    "ask_qtys": ask_qtys,
                }
            )

    return payload

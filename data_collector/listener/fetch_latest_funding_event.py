#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Optional

from binance_http import DEFAULT_INSECURE_SSL, http_get_with_retry

FAPI_BASE_DEFAULT = "https://fapi.binance.com"


def _parse_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def fetch_latest_funding_event(
    symbol: str,
    *,
    perp_base: str = FAPI_BASE_DEFAULT,
    user_agent: str = "arbitrage-listener/1.0",
    insecure_ssl: bool = DEFAULT_INSECURE_SSL,
    retries: int = 3,
    pause: float = 0.5,
    include_index_price: bool = False,
) -> Optional[Dict[str, object]]:
    symbol = symbol.strip().upper()

    data = http_get_with_retry(
        "/fapi/v1/fundingRate",
        {"symbol": symbol, "limit": 1},
        base=perp_base,
        user_agent=user_agent,
        insecure_ssl=insecure_ssl,
        retries=retries,
        pause=pause,
    )
    if not data:
        return None

    item = data[-1] if isinstance(data, list) else data
    event = {
        "symbol": symbol,
        "funding_time": int(item["fundingTime"]) if "fundingTime" in item else None,
        "funding_rate": _parse_float(item.get("fundingRate")),
        "mark_price": _parse_float(item.get("markPrice")),
        "index_price": None,
    }

    if include_index_price:
        idx = http_get_with_retry(
            "/fapi/v1/premiumIndex",
            {"symbol": symbol},
            base=perp_base,
            user_agent=user_agent,
            insecure_ssl=insecure_ssl,
            retries=retries,
            pause=pause,
        )
        if isinstance(idx, dict):
            event["index_price"] = _parse_float(idx.get("indexPrice"))
            if event["mark_price"] is None:
                event["mark_price"] = _parse_float(idx.get("markPrice"))

    return event

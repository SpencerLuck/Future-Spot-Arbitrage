#!/usr/bin/env python3
"""
Binance USDT-M PERPETUAL kline & funding downloader (importable + CLI).

Adds funding-rate history download per symbol:
- Resume-safe CSVs: <symbol>_funding.csv with columns [funding_time, funding_rate]
- Pagination via startTime/endTime, mirrors kline behavior
"""

from __future__ import annotations

import os
import csv
import time
import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict

# ---- Constants ---------------------------------------------------------------
FAPI_BASE_DEFAULT = "https://fapi.binance.com"
HEADERS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_volume","trade_count",
    "taker_base_volume","taker_quote_volume","ignore"
]
FUNDING_HEADERS = ["funding_time", "funding_rate"]
MS = 1000
NOW_MS = lambda: int(time.time() * 1000)

# ---- Optional requests import (fallback to urllib) ---------------------------
try:
    import requests  # type: ignore
    _HAVE_REQUESTS = True
except Exception:
    _HAVE_REQUESTS = False

# ---- Helpers -----------------------------------------------------------------
def interval_to_ms(interval: str) -> int:
    """Map Binance interval string (e.g., 1m, 1h, 1d) to milliseconds."""
    unit = interval[-1]
    num = int(interval[:-1])
    if unit == "m":
        return num * 60 * MS
    if unit == "h":
        return num * 60 * 60 * MS
    if unit == "d":
        return num * 24 * 60 * 60 * MS
    if unit == "w":
        return num * 7 * 24 * 60 * 60 * MS
    if unit == "M":  # calendar month; approximate stepping as 30 days
        return num * 30 * 24 * 60 * 60 * MS
    raise ValueError(f"Unsupported interval: {interval}")

def clamp_limit(limit: int, *, min_value: int, max_value: int) -> int:
    return max(min_value, min(limit, max_value))

def csv_path_for(symbol: str, interval: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{symbol}_{interval}_PERP.csv")

def funding_csv_path_for(symbol: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{symbol}_funding.csv")

def read_last_open_time(path: str) -> Optional[int]:
    """Read last non-empty, non-header line's open_time from CSV (fast tail scan)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        chunk = 128 * 1024
        read = min(size, chunk)
        f.seek(-read, os.SEEK_END)
        data = f.read(read)
    lines = [ln for ln in data.splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.startswith(b"open_time"):
            continue
        parts = line.split(b",")
        try:
            return int(parts[0])
        except Exception:
            continue
    return None

def read_last_funding_time(path: str) -> Optional[int]:
    """Read last funding_time from funding CSV."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        chunk = 64 * 1024
        read = min(size, chunk)
        f.seek(-read, os.SEEK_END)
        data = f.read(read)
    lines = [ln for ln in data.splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.startswith(b"funding_time"):
            continue
        parts = line.split(b",")
        try:
            return int(parts[0])
        except Exception:
            continue
    return None

def ensure_header(path: str):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    if new_file:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADERS)

def ensure_funding_header(path: str):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    if new_file:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FUNDING_HEADERS)

def save_csv_streaming(path: str, row_iter: Iterable[list]) -> int:
    """Append rows to kline CSV; returns number of rows written."""
    ensure_header(path)
    count = 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for r in row_iter:
            # r is Binance kline array (12 fields)
            w.writerow([r[0], r[1], r[2], r[3], r[4], r[5],
                        r[6], r[7], r[8], r[9], r[10], r[11]])
            count += 1
    return count

def save_funding_csv_streaming(path: str, row_iter: Iterable[list]) -> int:
    """Append rows to funding CSV; returns number of rows written."""
    ensure_funding_header(path)
    count = 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for r in row_iter:
            # r is [funding_time_ms, funding_rate_str]
            w.writerow([r[0], r[1]])
            count += 1
    return count

# ---- HTTP layer --------------------------------------------------------------
def http_get(
    path: str,
    params: Optional[dict],
    *,
    base: str,
    user_agent: str,
    insecure_ssl: bool,
    timeout: int = 30
):
    url = base + path
    params = params or {}

    if _HAVE_REQUESTS:
        verify: bool | str = True
        if insecure_ssl:
            verify = False
        elif os.environ.get("REQUESTS_CA_BUNDLE"):
            verify = os.environ["REQUESTS_CA_BUNDLE"]
        r = requests.get(url, params=params, timeout=timeout, verify=verify,
                         headers={"User-Agent": user_agent})
        r.raise_for_status()
        return r.json()

    # urllib fallback with certifi if available
    import ssl
    import urllib.parse
    import urllib.request

    qs = urllib.parse.urlencode(params)
    full_url = url + ("?" + qs if qs else "")
    req = urllib.request.Request(full_url, headers={"User-Agent": user_agent})

    if insecure_ssl:
        ctx = ssl._create_unverified_context()
    else:
        try:
            import certifi  # type: ignore
            ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or certifi.where())
        except Exception:
            ctx = ssl.create_default_context()

    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))

def http_get_with_retry(
    path: str,
    params: dict,
    *,
    base: str,
    user_agent: str,
    insecure_ssl: bool,
    retries: int,
    pause: float
):
    """Simple retry/backoff to be nice with transient issues."""
    delay = pause
    for attempt in range(1, retries + 1):
        try:
            return http_get(path, params, base=base, user_agent=user_agent, insecure_ssl=insecure_ssl)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 5.0)

# ---- Binance helpers ---------------------------------------------------------
def get_perpetual_symbols(
    *,
    base: str,
    user_agent: str,
    insecure_ssl: bool,
    retries: int,
    pause: float
) -> List[str]:
    """Return list of USDT-M PERPETUAL symbols that are TRADING (e.g., BTCUSDT)."""
    data = http_get_with_retry(
        "/fapi/v1/exchangeInfo",
        {},
        base=base, user_agent=user_agent, insecure_ssl=insecure_ssl,
        retries=retries, pause=pause
    )
    symbols = []
    for s in data.get("symbols", []):
        if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING":
            symbols.append(s["symbol"])
    return sorted(symbols)

def iter_klines_range(
    symbol: str,
    *,
    start_ms: int,
    end_ms: Optional[int],
    interval: str,
    limit: int,
    pause: float,
    base: str,
    user_agent: str,
    insecure_ssl: bool,
    retries: int
) -> Iterable[list]:
    """
    Yield klines from start_ms (inclusive) to end_ms (exclusive) in chronological order,
    fetching in pages of `limit` (max 1500). If end_ms is None, fetch until 'now'.
    """
    limit = clamp_limit(limit, min_value=1, max_value=1500)
    cur = int(start_ms)
    end_ms_int = int(end_ms) if end_ms is not None else None
    step_ms = interval_to_ms(interval)

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": cur,
        }
        if end_ms_int is not None:
            params["endTime"] = end_ms_int

        batch = http_get_with_retry(
            "/fapi/v1/klines", params,
            base=base, user_agent=user_agent, insecure_ssl=insecure_ssl,
            retries=retries, pause=pause
        )
        if not batch:
            break

        # Yield in chronological order
        for row in batch:
            yield row

        last_open = batch[-1][0]
        cur = last_open + step_ms

        # Stop if reached end or Binance indicates no more (returned < limit)
        if (end_ms_int is not None and cur > end_ms_int) or len(batch) < limit:
            break

        time.sleep(pause)

def iter_funding_range(
    symbol: str,
    *,
    start_ms: int,
    end_ms: Optional[int],
    limit: int,
    pause: float,
    base: str,
    user_agent: str,
    insecure_ssl: bool,
    retries: int
) -> Iterable[list]:
    """
    Yield funding history rows [funding_time_ms, funding_rate_str] for `symbol`
    between start_ms (inclusive) and end_ms (exclusive), paginating on `limit` (<=1000).
    """
    limit = clamp_limit(limit, min_value=1, max_value=1000)
    cur = int(start_ms)
    end_ms_int = int(end_ms) if end_ms is not None else None

    while True:
        params = {
            "symbol": symbol,
            "limit": limit,
            "startTime": cur,
        }
        if end_ms_int is not None:
            params["endTime"] = end_ms_int

        batch = http_get_with_retry(
            "/fapi/v1/fundingRate", params,
            base=base, user_agent=user_agent, insecure_ssl=insecure_ssl,
            retries=retries, pause=pause
        )
        if not batch:
            break

        # Ensure chronological order by fundingTime
        batch.sort(key=lambda x: int(x.get("fundingTime", 0)))

        # Emit rows
        for item in batch:
            ft = int(item["fundingTime"])
            fr = item["fundingRate"]  # string per Binance API
            yield [ft, fr]

        cur = int(batch[-1]["fundingTime"]) + 1

        # Stop if reached end or Binance indicates no more (returned < limit)
        if (end_ms_int is not None and cur > end_ms_int) or len(batch) < limit:
            break

        time.sleep(pause)

# ---- Public callable ---------------------------------------------------------
@dataclass
class FetchConfig:
    # selection
    symbols: Optional[List[str]] = None          # if None or empty, auto-discover PERPETUAL symbols
    interval: str = "1m"
    limit: int = 1500

    # time window
    lookback_minutes: int = 60 * 24 * 30         # used only if start_ms is None
    start_ms: Optional[int] = None               # if set, takes precedence over lookback
    end_ms: Optional[int] = None                 # default: NOW

    # io + behavior
    out_dir: str = os.path.join("arbitrage", "data")
    resume: bool = True
    pause: float = 0.12
    pause_between_symbols: float = 0.25
    retries: int = 5
    user_agent: str = "binance-perps-1m/0.3"
    base_url: str = FAPI_BASE_DEFAULT
    insecure_ssl: bool = False                   # skip TLS verification (not recommended)
    verbose: bool = True

    # funding extras
    with_funding: bool = False
    funding_limit: int = 1000

    # allow env defaults when fields are None / not overridden
    use_env_defaults: bool = True

    def apply_env_defaults(self):
        if not self.use_env_defaults:
            return
        env = os.environ

        self.interval = env.get("INTERVAL", self.interval)
        self.lookback_minutes = int(env.get("LOOKBACK_MINUTES", self.lookback_minutes))
        self.start_ms = int(env["START_MS"]) if env.get("START_MS") else self.start_ms
        self.end_ms = int(env["END_MS"]) if env.get("END_MS") else self.end_ms
        self.resume = (env.get("RESUME", "1") != "0") if self.resume is None else self.resume  # default on
        syms_env = env.get("SYMBOLS")
        if syms_env and (self.symbols is None or len(self.symbols) == 0):
            self.symbols = [x.strip().upper() for x in syms_env.split(",") if x.strip()]
        self.out_dir = env.get("OUT_DIR", self.out_dir)
        self.pause = float(env.get("PAUSE", self.pause))
        self.pause_between_symbols = float(env.get("PAUSE_BETWEEN_SYMBOLS", self.pause_between_symbols))
        self.retries = int(env.get("RETRIES", self.retries))
        self.user_agent = env.get("UA", self.user_agent)
        if env.get("INSECURE_SSL") == "1":
            self.insecure_ssl = True

        if env.get("WITH_FUNDING") in ("1", "true", "TRUE", "yes", "YES"):
            self.with_funding = True
        self.funding_limit = int(env.get("FUNDING_LIMIT", self.funding_limit))

def fetch_perpetual_klines(
    symbols: Optional[List[str]] = None,
    out_dir: str = os.path.join("arbitrage", "data"),
    interval: str = "1m",
    lookback_minutes: int = 60 * 24 * 30,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    resume: bool = True,
    pause: float = 0.12,
    pause_between_symbols: float = 0.25,
    retries: int = 5,
    user_agent: str = "binance-perps-1m/0.3",
    base_url: str = FAPI_BASE_DEFAULT,
    insecure_ssl: bool = False,
    limit: int = 1500,
    verbose: bool = True,
    use_env_defaults: bool = True,
    with_funding: bool = False,
    funding_limit: int = 1000,
) -> Dict[str, int]:
    """
    Download klines for given USDT-M PERPETUAL symbols to CSVs (+ optional funding CSVs).

    Returns: dict {symbol: rows_written_for_klines}
    """
    cfg = FetchConfig(
        symbols=list(symbols) if symbols else None,
        out_dir=out_dir,
        interval=interval,
        lookback_minutes=lookback_minutes,
        start_ms=start_ms,
        end_ms=end_ms,
        resume=resume,
        pause=pause,
        pause_between_symbols=pause_between_symbols,
        retries=retries,
        user_agent=user_agent,
        base_url=base_url,
        insecure_ssl=insecure_ssl,
        limit=limit,
        verbose=verbose,
        use_env_defaults=use_env_defaults,
        with_funding=with_funding,
        funding_limit=funding_limit,
    )
    cfg.apply_env_defaults()

    step_ms = interval_to_ms(cfg.interval)
    end_ms_eff = cfg.end_ms if cfg.end_ms is not None else NOW_MS()
    start_ms_eff = cfg.start_ms if cfg.start_ms is not None else (end_ms_eff - cfg.lookback_minutes * 60 * 1000)

    # Resolve symbols
    if not cfg.symbols:
        if cfg.verbose:
            print("Discovering USDT-M PERPETUAL symbols...")
        cfg.symbols = get_perpetual_symbols(
            base=cfg.base_url, user_agent=cfg.user_agent, insecure_ssl=cfg.insecure_ssl,
            retries=cfg.retries, pause=cfg.pause
        )

    if cfg.verbose:
        print(f"Found {len(cfg.symbols)} USDT-M perpetual symbols")
        print(f"Interval: {cfg.interval} (step {step_ms//1000}s)")
        lb_txt = "n/a; START_MS set" if cfg.start_ms is not None else f"{cfg.lookback_minutes}"
        print(f"Global default range: start_ms={start_ms_eff} end_ms={end_ms_eff} (LOOKBACK_MINUTES={lb_txt})")
        print(f"Resume mode: {'ON' if cfg.resume else 'OFF'}")
        print(f"Output dir: {cfg.out_dir}")
        if cfg.with_funding:
            print(f"Funding download: ON (limit={cfg.funding_limit})")

    results: Dict[str, int] = {}

    for i, sym in enumerate(cfg.symbols, start=1):
        try:
            # -------- KLINES --------
            path = csv_path_for(sym, cfg.interval, cfg.out_dir)

            # Compute start per-symbol (klines)
            start_ms_sym = start_ms_eff
            if cfg.resume and os.path.exists(path):
                last_ms = read_last_open_time(path)
                if last_ms is not None:
                    start_ms_sym = last_ms + step_ms  # continue after last candle

            end_ms_sym = end_ms_eff

            if end_ms_sym is not None and start_ms_sym is not None and start_ms_sym >= end_ms_sym:
                if cfg.verbose:
                    print(f"[{i}/{len(cfg.symbols)}] {sym}: up-to-date (klines), skipping")
                rows_written = 0
            else:
                if cfg.verbose:
                    print(f"[{i}/{len(cfg.symbols)}] {sym}: fetching klines {start_ms_sym}..{end_ms_sym} -> {path}")
                rows_written = save_csv_streaming(
                    path,
                    iter_klines_range(
                        sym,
                        start_ms=start_ms_sym,
                        end_ms=end_ms_sym,
                        interval=cfg.interval,
                        limit=cfg.limit,
                        pause=cfg.pause,
                        base=cfg.base_url,
                        user_agent=cfg.user_agent,
                        insecure_ssl=cfg.insecure_ssl,
                        retries=cfg.retries,
                    ),
                )
                if cfg.verbose:
                    print(f"[{i}/{len(cfg.symbols)}] {sym}: wrote {rows_written} kline rows")

            results[sym] = rows_written

            # -------- FUNDING --------
            if cfg.with_funding:
                fpath = funding_csv_path_for(sym, cfg.out_dir)

                f_start = start_ms_eff
                if cfg.resume and os.path.exists(fpath):
                    last_ft = read_last_funding_time(fpath)
                    if last_ft is not None:
                        f_start = last_ft + 1  # next ms after last funding event

                f_end = end_ms_eff
                if f_end is not None and f_start is not None and f_start >= f_end:
                    if cfg.verbose:
                        print(f"[{i}/{len(cfg.symbols)}] {sym}: up-to-date (funding), skipping")
                else:
                    if cfg.verbose:
                        print(f"[{i}/{len(cfg.symbols)}] {sym}: fetching funding {f_start}..{f_end} -> {fpath}")
                    f_rows = save_funding_csv_streaming(
                        fpath,
                        iter_funding_range(
                            sym,
                            start_ms=f_start,
                            end_ms=f_end,
                            limit=cfg.funding_limit,
                            pause=cfg.pause,
                            base=cfg.base_url,
                            user_agent=cfg.user_agent,
                            insecure_ssl=cfg.insecure_ssl,
                            retries=cfg.retries,
                        ),
                    )
                    if cfg.verbose:
                        print(f"[{i}/{len(cfg.symbols)}] {sym}: wrote {f_rows} funding rows")

            time.sleep(cfg.pause_between_symbols)

        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception as e:
            print(f"[{i}/{len(cfg.symbols)}] ERROR {sym}: {e}")
            time.sleep(1.0)
            results[sym] = results.get(sym, 0)

    return results

# ---- CLI ---------------------------------------------------------------------
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Download Binance USDT-M PERPETUAL klines (and funding) to CSV.")
    p.add_argument("--symbols", type=str, default=None,
                   help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT). If omitted, auto-discover all PERPETUAL.")
    p.add_argument("--out-dir", type=str, default=os.path.join("arbitrage", "data"))
    p.add_argument("--interval", type=str, default="1m")
    p.add_argument("--lookback-minutes", type=int, default=60*24*30)
    p.add_argument("--start-ms", type=int, default=None)
    p.add_argument("--end-ms", type=int, default=None)
    p.add_argument("--no-resume", action="store_true", help="Disable resume-from-CSV behavior.")
    p.add_argument("--pause", type=float, default=0.12)
    p.add_argument("--pause-between-symbols", type=float, default=0.25)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--user-agent", type=str, default="binance-perps-1m/0.3")
    p.add_argument("--base-url", type=str, default=FAPI_BASE_DEFAULT)
    p.add_argument("--insecure-ssl", action="store_true", help="Disable TLS verification (NOT recommended).")
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--quiet", action="store_true", help="Less verbose output.")
    p.add_argument("--no-env-defaults", action="store_true", help="Do not read env var defaults.")
    p.add_argument("--with-funding", action="store_true", help="Also download funding rates to separate CSVs.")
    p.add_argument("--funding-limit", type=int, default=1000, help="Funding history page size (<=1000).")
    return p.parse_args()

def main():
    args = _parse_args()
    syms = None if not args.symbols else [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    fetch_perpetual_klines(
        symbols=syms,
        out_dir=args.out_dir,
        interval=args.interval,
        lookback_minutes=args.lookback_minutes,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        resume=(not args.no_resume),
        pause=args.pause,
        pause_between_symbols=args.pause_between_symbols,
        retries=args.retries,
        user_agent=args.user_agent,
        base_url=args.base_url,
        insecure_ssl=args.insecure_ssl,
        limit=args.limit,
        verbose=(not args.quiet),
        use_env_defaults=(not args.no_env_defaults),
        with_funding=args.with_funding,
        funding_limit=args.funding_limit,
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse
from typing import Set, Tuple, Optional

# --- your http_get helper (unchanged) ----------------------------------------
try:
    import requests  # type: ignore
    _HAVE_REQUESTS = True
except Exception:
    _HAVE_REQUESTS = False

def http_get(
    path: str,
    params: dict | None,
    *,
    base: str,
    user_agent: str,
    insecure_ssl: bool,
    timeout: int = 30
):
    import json as _json
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
        return _json.loads(resp.read().decode("utf-8"))

# --- constants ----------------------------------------------------------------
FAPI_BASE = "https://fapi.binance.com"   # USDT-M futures
SPOT_BASE = "https://api.binance.com"    # Spot
UA = "altcoin-list/1.0"

EXCLUDE = {"BTCUSDT", "ETHUSDT"}
EXCLUDE_STABLES = {
    "USDCUSDT","FDUSDUSDT","TUSDUSDT","USDPUSDT","DAIUSDT","BUSDUSDT",
    "EURUSDT","TRYUSDT","RUBUSDT","UAHUSDT","BRLUSDT"
}

# --- helpers ------------------------------------------------------------------
def get_usdtm_perp_symbols(insecure_ssl: bool) -> Set[str]:
    data = http_get("/fapi/v1/exchangeInfo", None, base=FAPI_BASE, user_agent=UA, insecure_ssl=insecure_ssl)
    return {
        s["symbol"]
        for s in data.get("symbols", [])
        if s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
    }

def get_spot_usdt_symbols(insecure_ssl: bool) -> Set[str]:
    data = http_get("/api/v3/exchangeInfo", None, base=SPOT_BASE, user_agent=UA, insecure_ssl=insecure_ssl)
    return {
        s["symbol"]
        for s in data.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
    }

def normalized_spot_symbol(perp_sym: str, spot_set: Set[str]) -> Optional[str]:
    """
    Try to map multiplier futures like '1000PEPEUSDT' -> 'PEPEUSDT' if exact spot symbol doesn't exist.
    This is heuristic (strip leading digits from base asset) and may not always be correct.
    """
    if perp_sym in spot_set:
        return perp_sym
    if perp_sym.endswith("USDT"):
        base = perp_sym[:-4]
        base2 = base.lstrip("0123456789")
        candidate = f"{base2}USDT"
        if candidate in spot_set:
            return candidate
    return None

def build_alt_lists(insecure_ssl: bool, exclude_btc_eth: bool = True, exclude_stables: bool = True
                   ) -> Tuple[list[str], list[Tuple[str, str]]]:
    perps = get_usdtm_perp_symbols(insecure_ssl)
    spot  = get_spot_usdt_symbols(insecure_ssl)

    # filter excludes
    drop = set()
    if exclude_btc_eth:
        drop |= EXCLUDE
    if exclude_stables:
        drop |= EXCLUDE_STABLES

    # exact matches
    exact = sorted(s for s in (perps & spot) if s not in drop)

    # normalized matches for multiplier perps whose spot symbol differs
    norm_pairs = []
    for p in sorted(perps - set(exact)):
        if p in drop:
            continue
        sp = normalized_spot_symbol(p, spot)
        if sp and sp not in drop and p != sp:
            norm_pairs.append((p, sp))

    return exact, norm_pairs

# --- CLI ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="List Binance USDT-M altcoins that also have SPOT USDT pairs.")
    ap.add_argument("--secure", action="store_true", help="Verify TLS certs (default is INSECURE).")
    ap.add_argument("--include-btc-eth", action="store_true", help="Include BTCUSDT and ETHUSDT.")
    ap.add_argument("--include-stables", action="store_true", help="Include stable-coin pairs.")
    args = ap.parse_args()

    # default to insecure unless overridden via flag or env
    env_insecure = os.environ.get("INSECURE_SSL", "1")  # "1" by default
    insecure_ssl = (env_insecure != "0") and (not args.secure)

    exact, normalized = build_alt_lists(
        insecure_ssl=insecure_ssl,
        exclude_btc_eth=not args.include_btc_eth,
        exclude_stables=not args.include_stables
    )

    print(f"# insecure_ssl = {insecure_ssl}")
    print(f"# EXACT matches (perp symbol also exists on spot): {len(exact)}")
    print("SYMBOLS = [")
    for s in exact:
        print(f'    "{s}",')
    print("]\n")

    if normalized:
        print(f"# NORMALIZED matches (perp -> spot) where naming differs: {len(normalized)}")
        print("# Use these as (PERP_SYMBOL, SPOT_SYMBOL) pairs if your pipeline supports mapping.")
        for p, sp in normalized:
            print(f'#   {p} -> {sp}')

if __name__ == "__main__":
    main()
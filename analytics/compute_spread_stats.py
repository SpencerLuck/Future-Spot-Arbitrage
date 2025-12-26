from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence, Tuple
from urllib.parse import quote, urlparse, urlunparse

import pandas as pd
import psycopg2
from psycopg2 import sql


def _load_env_file(path: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or key in os.environ:
                    continue
                if (
                    len(value) >= 2
                    and value[0] == value[-1]
                    and value[0] in {"'", "\""}
                ):
                    value = value[1:-1]
                os.environ[key] = value
    except FileNotFoundError:
        return


def _running_in_docker() -> bool:
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def _normalize_dsn_for_local(dsn: str) -> str:
    if "://" not in dsn:
        return dsn
    parsed = urlparse(dsn)
    if parsed.hostname != "db" or _running_in_docker():
        return dsn
    user = parsed.username or ""
    password = parsed.password or ""
    host = "localhost"
    port = parsed.port
    auth = ""
    if user:
        auth = quote(user, safe="")
        if password:
            auth += f":{quote(password, safe='')}"
        auth += "@"
    netloc = f"{auth}{host}"
    if port:
        netloc += f":{port}"
    return urlunparse(
        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def _parse_symbols(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _build_query(
    schema: str,
    symbols: Sequence[str],
    since_ms: Optional[int],
    until_ms: Optional[int],
) -> Tuple[str, List[object]]:
    conditions = [
        "market_type in ('spot', 'perp')",
        "bid_price is not null",
        "ask_price is not null",
    ]
    params: List[object] = []
    if symbols:
        conditions.append("symbol = any(%s)")
        params.append(list(symbols))
    if since_ms is not None:
        conditions.append("ts_ms >= %s")
        params.append(since_ms)
    if until_ms is not None:
        conditions.append("ts_ms < %s")
        params.append(until_ms)

    where_sql = " and ".join(conditions)

    query = sql.SQL(
        """
        with base as (
            select
                symbol,
                market_type,
                (ts_ms / 1000) * 1000 as t_ms,
                ts_ms,
                bid_price,
                ask_price
            from {schema}.quote_snapshots
            where {where_clause}
        ),
        latest as (
            select distinct on (symbol, market_type, t_ms)
                symbol,
                market_type,
                t_ms,
                bid_price,
                ask_price
            from base
            order by symbol, market_type, t_ms, ts_ms desc
        ),
        spot as (
            select
                symbol,
                t_ms,
                bid_price as s_bid,
                ask_price as s_ask
            from latest
            where market_type = 'spot'
        ),
        perp as (
            select
                symbol,
                t_ms,
                bid_price as p_bid,
                ask_price as p_ask
            from latest
            where market_type = 'perp'
        )
        select
            spot.symbol,
            spot.t_ms,
            spot.s_bid,
            spot.s_ask,
            perp.p_bid,
            perp.p_ask
        from spot
        join perp
            on perp.symbol = spot.symbol
           and perp.t_ms = spot.t_ms
        order by spot.symbol, spot.t_ms
        """
    ).format(
        schema=sql.Identifier(schema),
        where_clause=sql.SQL(where_sql),
    )

    return query, params


def _compute_series(df: pd.DataFrame) -> pd.DataFrame:
    series = df.copy()
    series["s_mid"] = (series["s_bid"] + series["s_ask"]) / 2.0
    series["p_mid"] = (series["p_bid"] + series["p_ask"]) / 2.0
    series["s_spread"] = series["s_ask"] - series["s_bid"]
    series["p_spread"] = series["p_ask"] - series["p_bid"]
    series["spread_entry"] = series["p_bid"] - series["s_ask"]
    series["spread_exit"] = series["p_ask"] - series["s_bid"]
    series["spread_entry_bps"] = (series["spread_entry"] / series["s_mid"]) * 10000.0
    series["spread_exit_bps"] = (series["spread_exit"] / series["s_mid"]) * 10000.0
    series["s_spread_bps"] = (series["s_spread"] / series["s_mid"]) * 10000.0
    series["p_spread_bps"] = (series["p_spread"] / series["s_mid"]) * 10000.0
    return series


def _compute_stats(series: pd.DataFrame) -> pd.DataFrame:
    grouped = series.groupby("symbol", sort=True)

    stats = grouped["spread_entry_bps"].agg(
        spread_entry_bps_mean="mean",
        spread_entry_bps_median="median",
        spread_entry_bps_p95=lambda x: x.quantile(0.95),
        spread_entry_bps_p99=lambda x: x.quantile(0.99),
    )

    stats["s_spread_bps_mean"] = grouped["s_spread_bps"].mean()
    stats["p_spread_bps_mean"] = grouped["p_spread_bps"].mean()

    return stats.reset_index()


def main() -> None:
    _load_env_file(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(
        description="Compute spread analytics from quote snapshots."
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DB_DSN"),
        help="Postgres DSN (default: DB_DSN env var)",
    )
    parser.add_argument(
        "--schema",
        default=os.environ.get("DB_SCHEMA", "market_data"),
        help="DB schema containing quote_snapshots",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol list (default: all)",
    )
    parser.add_argument(
        "--since-ms",
        type=int,
        default=None,
        help="Filter ts_ms >= since-ms",
    )
    parser.add_argument(
        "--until-ms",
        type=int,
        default=None,
        help="Filter ts_ms < until-ms",
    )
    parser.add_argument(
        "--out-series",
        default=None,
        help="Optional CSV path to write the time series",
    )
    parser.add_argument(
        "--out-stats",
        default=None,
        help="Optional CSV path to write summary stats",
    )

    args = parser.parse_args()

    if not args.dsn:
        raise SystemExit("DB_DSN is required (set env var or --dsn).")

    dsn = _normalize_dsn_for_local(args.dsn)
    symbols = _parse_symbols(args.symbols)

    conn = psycopg2.connect(dsn)
    try:
        query, params = _build_query(args.schema, symbols, args.since_ms, args.until_ms)
        query_sql = query.as_string(conn)
        df = pd.read_sql(query_sql, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        raise SystemExit("No rows returned for the requested filters.")

    series = _compute_series(df)
    stats = _compute_stats(series)

    if args.out_series:
        series.to_csv(args.out_series, index=False)
    if args.out_stats:
        stats.to_csv(args.out_stats, index=False)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()

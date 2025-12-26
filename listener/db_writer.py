from __future__ import annotations

from typing import Any, Dict, Optional

import psycopg2
from psycopg2 import sql


class DbWriter:
    def __init__(self, dsn: str, schema: str = "market_data") -> None:
        self.dsn = dsn
        self.schema = schema
        self.conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        if self.conn is not None and self.conn.closed == 0:
            return
        self.conn = psycopg2.connect(self.dsn)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("set search_path to {}").format(
                    sql.Identifier(self.schema)
                )
            )

    def insert_quote(self, snapshot: Dict[str, Any]) -> None:
        self.connect()
        if self.conn is None:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                """
                insert into quote_snapshots (
                    ts_ms,
                    exchange_ts_ms,
                    symbol,
                    market_type,
                    bid_price,
                    bid_qty,
                    ask_price,
                    ask_qty,
                    bid_prices,
                    bid_qtys,
                    ask_prices,
                    ask_qtys,
                    iter_id
                )
                values (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    snapshot.get("ts"),
                    snapshot.get("exchange_ts"),
                    snapshot.get("symbol"),
                    snapshot.get("market_type"),
                    snapshot.get("bid_price"),
                    snapshot.get("bid_qty"),
                    snapshot.get("ask_price"),
                    snapshot.get("ask_qty"),
                    snapshot.get("bid_prices"),
                    snapshot.get("bid_qtys"),
                    snapshot.get("ask_prices"),
                    snapshot.get("ask_qtys"),
                    snapshot.get("iter_id"),
                ),
            )

    def insert_funding_event(self, event: Dict[str, Any]) -> None:
        self.connect()
        if self.conn is None:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                """
                insert into funding_events (
                    symbol,
                    funding_time_ms,
                    funding_rate,
                    mark_price,
                    index_price
                )
                values (%s, %s, %s, %s, %s)
                on conflict (symbol, funding_time_ms) do nothing
                """,
                (
                    event.get("symbol"),
                    event.get("funding_time"),
                    event.get("funding_rate"),
                    event.get("mark_price"),
                    event.get("index_price"),
                ),
            )

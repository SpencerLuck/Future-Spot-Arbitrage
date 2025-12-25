-- Postgres schema + tables for Binance listener market data.

create schema if not exists market_data;

set search_path to market_data;

create table if not exists quote_snapshots (
    id bigserial primary key,
    ts_ms bigint not null,
    exchange_ts_ms bigint,
    symbol text not null,
    market_type text not null,
    bid_price double precision,
    bid_qty double precision,
    ask_price double precision,
    ask_qty double precision,
    bid_prices double precision[],
    bid_qtys double precision[],
    ask_prices double precision[],
    ask_qtys double precision[],
    iter_id bigint,
    inserted_at timestamptz not null default now(),
    check (market_type in ('spot', 'perp'))
);

create index if not exists quote_snapshots_symbol_mt_ts_idx
    on quote_snapshots (symbol, market_type, ts_ms desc);

create index if not exists quote_snapshots_exchange_ts_idx
    on quote_snapshots (exchange_ts_ms);

create table if not exists funding_events (
    id bigserial primary key,
    symbol text not null,
    funding_time_ms bigint not null,
    funding_rate double precision,
    mark_price double precision,
    index_price double precision,
    inserted_at timestamptz not null default now()
);

create unique index if not exists funding_events_symbol_time_uq
    on funding_events (symbol, funding_time_ms);

create index if not exists funding_events_time_idx
    on funding_events (funding_time_ms desc);

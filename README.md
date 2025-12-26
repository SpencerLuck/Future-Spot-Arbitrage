# Perp-Spot Arbitrage Strategy & Data Collection (Listener)

This repo backtests a simple spot-perp arbitrage strategy on Binance data. The strategy looks for times when perpetual futures trade above spot by more than costs, then enters a market-neutral hedge: short perp, long spot.

## Strategy Overview

- Signal: `spread = perp_price - spot_price`
- Threshold: fee-only or fee-minus-funding [net] (configurable)
- Entry: when `spread > threshold` (and optionally `spread > 0`)
- Exit: when `spread` falls back below the threshold
- Position: long spot + short perp (same quantity)

Costs and carry:
- Slippage: applied per leg on entry and exit
- Taker fees: applied per leg on entry and exit
- Funding: accrued over the holding window (perp funding rates)


## Listener (Docker)

Edit `.env`, then start the stack:

```bash
docker compose up -d
```

Follow logs:

```bash
docker compose logs -f listener
```

Stop/start just the listener:

```bash
docker compose stop listener
docker compose start listener
```

Tear everything down:

```bash
docker compose down
```

### Environment Variables

These are all defined in `.env` and loaded by Docker Compose.

- `POSTGRES_DB`: database name for the Postgres container.
- `POSTGRES_USER`: database user for the Postgres container.
- `POSTGRES_PASSWORD`: database password for the Postgres container.
- `DB_DSN`: Postgres DSN for the listener (if unset, it only prints JSON).
- `DB_SCHEMA`: schema name for the listener tables (default `market_data`).
- `SYMBOLS`: comma-separated list like `BTCUSDT,ETHUSDT`.
- `MARKET_TYPE`: `spot`, `perp`, or `both`.
- `QUOTE_INTERVAL_MS`: poll interval for quote snapshots.
- `QUOTE_DEPTH_LEVELS`: number of depth levels; `0` disables depth arrays.
- `USER_AGENT`: HTTP User-Agent for API requests.
- `INSECURE_SSL`: `true` to skip SSL verification, `false` to enforce.
- `RETRIES`: number of HTTP retry attempts.
- `PAUSE`: seconds to wait between retries.
- `WITH_FUNDING`: `true` to poll funding events, `false` to skip.
- `FUNDING_INTERVAL_MS`: poll interval for funding events.
- `INCLUDE_INDEX_PRICE`: `true` to include Binance futures index price.

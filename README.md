# Perp-Spot Arbitrage Strategy & Backtester

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

## Backtester Execution Model

- Per-symbol backtest (no portfolio overlap)
- Enter on the next bar after a signal window starts
- Exit on the next bar after the window ends
- Position sizing: percent of equity, allocated to the two-leg notional
- Equity updated after each trade; summary stats computed from trade list

## Inputs

CSVs are expected in:
- Perp: `<perp_dir>/<SYMBOL>_<interval>_PERP.csv` and `<perp_dir>/<SYMBOL>_funding.csv`
- Spot: `<spot_dir>/<SYMBOL>_<interval>_SPOT.csv`

By default, `perp_dir` and `spot_dir` match the ingest scripts:
- `arbitrage/data`
- `arbitrage/data_spot`

## Outputs

Written once at the end of the run:
- `data/arb_trades/ALL_TRADES.csv`
- `data/arb_trades/ALL_SUMMARY.csv`

If `cleanup_raw_after_backtest = True`, raw CSVs for each symbol are deleted after its backtest completes.

## Configure & Run

Edit `src/backtester_config.py`:
- `interval`, `start_date`, `end_date` (or `start_ms` / `end_ms`)
- `equity_pct`, `slippage_bps`, fees
- `run_download` (download each symbol before backtest)
- `cleanup_raw_after_backtest` (delete source CSVs after each symbol)
- `symbols` or `use_alt_coins` + `alt_coins_path`

Run:

```bash
python src/backtester.py
```

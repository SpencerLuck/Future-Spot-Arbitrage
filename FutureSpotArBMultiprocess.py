import ftx
from datetime import *
import numpy as np
import time
from ArbitrageUtils import ratios, stats


MIN_VOL = 1_000_000

def check_trade(bid_ask):
    target = {}
    for ratio in bid_ask:
        perp_bid = bid_ask[f'{ratio}']['perp_bid']
        perp_ask = bid_ask[f'{ratio}']['perp_ask']
        spot_ask = bid_ask[f'{ratio}']['spot_ask']
        spot_bid = bid_ask[f'{ratio}']['spot_bid']
        volume = bid_ask[f'{ratio}']['volume']
        if volume > MIN_VOL:
            if perp_bid > spot_ask and perp_ask > spot_bid:
                entry_spread = perp_bid - spot_ask
                exit_spread = perp_ask - spot_bid
                trade_threshold = 0.0028 * (perp_bid + spot_ask) + abs(stats[f'{ratio}']['mean']) + stats[f'{ratio}']['q3']
                future_stats = SignIn.get_future_stats(f'{ratio}-PERP')
                if entry_spread > trade_threshold and exit_spread > trade_threshold and future_stats['nextFundingRate'] > 0:
                    entry_excess = float(((entry_spread - trade_threshold) / trade_threshold) * 100)
                    exit_excess = float(((exit_spread - trade_threshold) / trade_threshold) * 100)
                    target[(entry_excess+exit_excess)/2] = f'{ratio}'
                else:
                    pass
            else:
                pass
        else:
            pass

    # returning the currency with max excess threshold
    if len(target) > 0:
        max_target = target[max(target.keys())]
        return max_target
    else:
        return None

def open_positions():
    open_positions = []
    orders = SignIn.get_positions()
    for order in orders:
        if order['size'] > 0:
            if order['future'].rsplit('-', -1)[-1] == 'PERP':
                open_positions.append(order)
    return open_positions

# current trade dictionary
current_trades = {}


# On Init
S_api_key = '511V4FYAfBEBSCi4AOH9FL4YRQTSSBs8SIBGdzFz'
S_api_secret = '8IQFj_oKryBQGl87UoJFi1QzFgOKLDMaZuJw-4rG'
SignIn = ftx.FtxClient(api_key=S_api_key, api_secret=S_api_secret)
AccInfo = SignIn.get_account_info()


# On tick
while True:

    sleep_trade = 1 - datetime.now().minute % 1

    if sleep_trade == 1:
        static_time = datetime.now()

        # Get current bid-ask data
        bid_ask = {}
        markets = SignIn.get_markets()
        for market in markets:
            for ratio in ratios:
                if market['name'] == f'{ratio}-PERP':
                    bid_ask[f'{ratio}'] = {'perp_bid': market['bid'], 'perp_ask': market['ask'], 'volume': market['volumeUsd24h']}
                if market['name'] == f'{ratio}/USD':
                    spot_ask, spot_bid = market['ask'], market['bid']
                    bid_ask[f'{ratio}']['spot_ask'], bid_ask[f'{ratio}']['spot_bid'] = spot_ask, spot_bid

        # Checking for open positions
        '''get_positions only contains the future positions as spots
        are buys or sells of the actual asset'''
        existing_order = open_positions()

        if len(existing_order) > 0:
            currency = existing_order[0]['future'].rsplit('-', -1)[0]
            current_spread = bid_ask[f'{currency}']['perp_ask'] - bid_ask[f'{currency}']['spot_bid']

            if float(current_spread) <= float(stats[f'{currency}']['q3']): # close positions, check for future trade opp

                open_order_market, open_order_size = existing_order[0]['future'], existing_order[0]['size']

                # close perp
                close_perp = SignIn.place_order(market=open_order_market, side='buy', type='market', price=0,
                                           size=open_order_size, reduce_only=True)

                # sell spot
                sell_spot = SignIn.place_order(market=f'{currency}/USD', side='sell', type='market', price=0,
                                               size=open_order_size)
                current_trades.clear()
                print(f'{static_time}: Trade closed', current_trades)


                # Check for trade opportunities and execute
                trade_target = check_trade(bid_ask)
                if trade_target is None:
                    print(f'{static_time}: No trade opportunity')
                else:
                    print(f'Placing trade on {trade_target}')
                    acc_info = SignIn.get_account_info()
                    acc_capital = (acc_info['freeCollateral']*0.8)/2
                    order_size = float(acc_capital /
                                       ((bid_ask[f'{trade_target}']['perp_bid'] + bid_ask[f'{trade_target}']['spot_ask'])/2))
                    # Short perp
                    sell_perp = SignIn.place_order(market=f'{trade_target}-PERP', side='sell', type='market', price=0,
                                                    size=order_size)
                    # Buy spot
                    buy_spot = SignIn.place_order(market=f'{trade_target}/USD', side='buy', type='market', price=0,
                                                    size=order_size)

                    current_trades[f'{trade_target}'] = {'PERP': sell_perp, 'SPOT': buy_spot}
                    print(f'{static_time}: Trade placed', current_trades)

            elif float(current_spread) > float(stats[f'{currency}']['q3']): # don't close position
                print(f'{static_time}: Position open, no exit signal', current_trades)

        # Check for trade opportunities
        else:
            trade_target = check_trade(bid_ask)
            if trade_target is None:
                print(f'{static_time}: No trade opportunity')
            else:
                print(f'Placing trade on {trade_target}')
                acc_info = SignIn.get_account_info()
                acc_capital = (acc_info['freeCollateral'] * 0.8) / 2
                order_size = float(acc_capital / (
                            (bid_ask[f'{trade_target}']['perp_bid'] + bid_ask[f'{trade_target}']['spot_ask']) / 2))
                # Sell perp
                sell_perp = SignIn.place_order(market=f'{trade_target}-PERP', side='sell', type='market', price=0,
                                               size=order_size)
                # Buy spot
                buy_spot = SignIn.place_order(market=f'{trade_target}/USD', side='buy', type='market', price=0,
                                              size=order_size)

                current_trades[f'{trade_target}'] = {'PERP': sell_perp, 'SPOT': buy_spot}

                print(f'{static_time}: Trade placed', current_trades)

        time.sleep(sleep_trade * 60)
    else:
        time.sleep(sleep_trade * 60)
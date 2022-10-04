# Future-Spot-Arbitrage

This is a unique arbitrage strategy developed to exploit mispricings between Crypto Currency perpetual futures and spot markets. This strategy was specifically developed on the FTX exchange, but should theoretically be applicable on other exchanges offering perpetual future and spot instruments.

### What is Arbitrage?
Arbitrage is the practice of exploiting an assets price differences in different markets. A simple example would be if product X was listed in Market A and Market B. This means we can buy and sell product X in both Market A and Market B. Additionally, product X is transferable meaning that we can buy it in Market A and sell it in Market B or vice versa. If product X was for sale for $1 in Market A and Market B was selling product X for $1.20. We would buy product X in the cheaper market, Market A and sell it in Market B. Our profit would be the difference between what we bought it for and what we sold it for, thus netting $0.20 per transaction (excluding other costs). This profit is technically risk free. This was a very simple example with the majority of arbitrage strategies being far more complex and containing some form of risk.

### Arbitrage between Spot and Future Markets
The word arbitrage is used loosely in this context as this strategy is far from what would be considered traditional abritrage however, considering that it is exploiting mispricings between two markets it may still be arbitrage-esque in a modern sense. The strategy operates on the same exchange and on the same cryptocurrency, but between two markets, spot markets and perpetual future markets. Spot markets allow the purchase and possession of the actual currency (eg: BTC) and track the market value of a coin. The limitation of such markets for retail traders is that they don't allow a trader to short sell the asset easily. Perpetual futures markets on the other hand are markets for a perpetual futures contract tracking the coin. They allow traders to take both long and short positions easily.

The theory underpinning this strategy is that the spot price and perpetual future price for a particular cryptocurrency should be very similar or the same as they are tracking the same asset. Additionally perpetual futures have a funding rate which is an hourly payment between long and short position holders. If the perpetual futures contracts are trading at a premium to the index price of the currency then long positions pay short positions. When contracts are trading at a discount then short positions pay long positions. This acts as a measure to maintain perpetual futures prices within a reasonable range of the index price and by extension the spot price.   

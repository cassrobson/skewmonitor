import alpaca_trade_api as tradeapi
import numpy as np
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)

symbol = "SPY"
nvda = Stock(symbol , "SMART", "USD")
ib.qualifyContracts(nvda)
[ticker] = ib.reqTickers(nvda)
nvdaValue = ticker.marketPrice()
chains  = ib.reqSecDefOptParams(nvda.symbol, "", nvda.secType, nvda.conId)
chain = next(c for c in chains if c.tradingClass == symbol  and c.exchange == "SMART")


strikes = [strike for strike in chain.strikes if strike % 5 == 0 and nvdaValue -20 < strike < nvdaValue + 20]
expirations = sorted(exp for exp in chain.expirations)[:3]
rights = ["C", "P"]

contracts = [Option(symbol , expiration, strike, right, "CBOE", tradingClass = symbol) for right in rights for expiration in expirations for strike in strikes]

contracts = ib.qualifyContracts(*contracts)
tickers = ib.reqTickers(*contracts)

print(tickers[0])
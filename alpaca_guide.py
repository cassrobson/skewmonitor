import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import ContractType
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest, OptionSnapshotRequest
from alpaca.data.requests import OptionChainRequest

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
option_historical_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

class Util:
    @staticmethod
    def to_dataframe(data):
        if isinstance(data, list):
            return pd.DataFrame([item.__dict__ for item in data])
        return pd.DataFrame(data, columns=['tag', 'value']).set_index('tag')

def get_options_chain(underlying_symbol, contract_type, expiration_date_gte=None, expiration_date_lte=None, expiration_date=None, strike_price_gte=None, strike_price_lte=None):
    option_contracts = []
    params = GetOptionContractsRequest(
        underlying_symbols = [underlying_symbol], 
        expiration_date=expiration_date, 
        expiration_date_gte=expiration_date_gte,
        expiration_date_lte=expiration_date_lte,
        strike_price_gte= str(strike_price_gte) if strike_price_gte else None,
        strike_price_lte=str(strike_price_lte) if strike_price_lte else None,
        type=contract_type,
        limit = 10000
    )

    options = trading_client.get_option_contracts(params)
    option_contracts.extend(options.option_contracts)

    while options.next_page_token:
        params.page_token = options.next_page_token
        options = trading_client.get_option_contracts(params)
        option_contracts.extend(options.option_contracts)
    
    return Util.to_dataframe(option_contracts) 

def get_options_symbol(underlying_symbol, expiry, strike, contract):
    assert contract in [ContractType.CALL, ContractType.PUT]
    df_options = get_options_chain(underlying_symbol=underlying_symbol, expiration_date=expiry, strike_price_gte=strike, strike_price_lte=strike, contract_type=contract)
    assert df_options.shape[0]== 1

    return df_options.iloc[0]['symbol']
#options_chain = get_options_chain('SPY', 'call', '2025-03-21', '2025-03-28', None, 590, 600)
option_symbol = get_options_symbol('SPY', '2025-03-21', strike=590, contract=ContractType.CALL)

# snapshot = option_historical_data_client.get_option_snapshot(
#     OptionSnapshotRequest(
#         symbol_or_symbols=option_symbol
#     )
# )

# print(snapshot)

chain_snap = option_historical_data_client.get_option_chain(
    OptionChainRequest(
        underlying_symbol='BF/B'
    )
)

data_list = []
for symbol, details in chain_snap.items():
    quote = details.latest_quote  # Access attributes directly
    trade = details.latest_trade
    greeks = details.greeks

    data_list.append({
        'symbol': symbol,
        'bid_price': quote.bid_price if quote else None,
        'ask_price': quote.ask_price if quote else None,
        'bid_size': quote.bid_size if quote else None,
        'ask_size': quote.ask_size if quote else None,
        'bid_exchange': quote.bid_exchange if quote else None,
        'ask_exchange': quote.ask_exchange if quote else None,
        'quote_timestamp': quote.timestamp if quote else None,
        'trade_price': trade.price if trade else None,
        'trade_size': trade.size if trade else None,
        'trade_exchange': trade.exchange if trade else None,
        'trade_timestamp': trade.timestamp if trade else None,
        'implied_volatility': details.implied_volatility,
        'delta': greeks.delta if greeks else None,
        'gamma': greeks.gamma if greeks else None,
        'theta': greeks.theta if greeks else None,
        'vega': greeks.vega if greeks else None,
        'rho': greeks.rho if greeks else None
    })

# Convert list to DataFrame
df = pd.DataFrame(data_list)

# Convert timestamps to readable format
df['quote_timestamp'] = pd.to_datetime(df['quote_timestamp']).dt.tz_localize(None)
df['trade_timestamp'] = pd.to_datetime(df['trade_timestamp']).dt.tz_localize(None)

df['expiry_date'] = df['symbol'].str[3:9]  # Correct slice (3rd to 8th index)
df['expiry_date'] = '20' + df['expiry_date']  # Convert YYMMDD to YYYYMMDD
df['expiry_date'] = pd.to_datetime(df['expiry_date'], format='%Y%m%d').dt.date  # Convert to YYYY-MM-DD

# Extract option type (C or P)
df['option_type'] = df['symbol'].str[9]

# Extract strike price, convert to float
df['strike_price'] = df['symbol'].str[10:].astype(int) / 1000  

# Display the cleaned DataFrame
print(df)


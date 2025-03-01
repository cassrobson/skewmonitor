import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")

from alpaca.data.requests import OptionLatestQuoteRequest, OptionSnapshotRequest, OptionChainRequest
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
stock_historical_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

option_historical_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

import warnings
import matplotlib.pyplot as plt
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from datetime import datetime

warnings.filterwarnings('ignore')

class Util:
    @staticmethod
    def to_dataframe(data):
        if isinstance(data, list):
            return pd.DataFrame([item.__dict__ for item in data])
        return pd.DataFrame(data, columns=['tag', 'value']).set_index('tag')
    
def parse_option_symbol(symbol):
    # Regular expression pattern to match option symbols
    pattern = r'([A-Z]+)(\d{6})([CP])(\d+)'
    match = re.match(pattern, symbol)
    
    if match:
        ticker, expiry, option_type, strike = match.groups()
        
        # Convert expiry to datetime
        expiry_date = pd.to_datetime('20' + expiry, format='%Y%m%d').date()
        
        # Convert strike price to float
        strike_price = float(strike) / 1000
        
        return ticker, expiry_date, option_type, strike_price
    else:
        return None, None, None, None

def get_chain_snapshot(symbol, spot, expiry):
    chain_snap = option_historical_data_client.get_option_chain(
    OptionChainRequest(
        underlying_symbol=symbol, 
        expiration_date=expiry,
        strike_price_gte=str(spot-5),
        strike_price_lte=str(spot+5),

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
    df[['ticker', 'expiry_date', 'option_type', 'strike_price']] = df['symbol'].apply(parse_option_symbol).apply(pd.Series)

    # Convert timestamps to readable format
    # df['quote_timestamp'] = pd.to_datetime(df['quote_timestamp']).dt.tz_localize(None)
    # df['trade_timestamp'] = pd.to_datetime(df['trade_timestamp']).dt.tz_localize(None)

    # df['expiry_date'] = df['symbol'].str[3:9]  # Correct slice (3rd to 8th index)
    # df['expiry_date'] = '20' + df['expiry_date']  # Convert YYMMDD to YYYYMMDD
    
    # df['expiry_date'] = pd.to_datetime(df['expiry_date'], format='%Y%m%d').dt.date  # Convert to YYYY-MM-DD

    # # Extract option type (C or P)
    # df['option_type'] = df['symbol'].str[9]

    # # Extract strike price, convert to float
    # df['strike_price'] = df['symbol'].str[10:].astype(int) / 1000 
    return df


def get_next_third_friday():
    today = datetime.today()
    
    # Determine the upcoming month
    if today.day > 14:  # If we're past the second Friday, go to next month
        target_month = today.month + 1
        target_year = today.year if target_month <= 12 else today.year + 1
        target_month = target_month if target_month <= 12 else 1
    else:
        target_month = today.month
        target_year = today.year
    
    # Find the third Friday of the target month
    first_day = datetime(target_year, target_month, 1)
    weekday_of_first = first_day.weekday()  # 0 = Monday, ..., 4 = Friday
    first_friday = 1 + (4 - weekday_of_first) % 7
    third_friday = first_friday + 14
    third_friday_date = datetime(target_year, target_month, third_friday)

    # If the third Friday is within 7 days, roll over to the next month's third Friday
    if (third_friday_date - today).days <= 7:
        target_month += 1
        if target_month > 12:
            target_month = 1
            target_year += 1

        first_day = datetime(target_year, target_month, 1)
        weekday_of_first = first_day.weekday()
        first_friday = 1 + (4 - weekday_of_first) % 7
        third_friday = first_friday + 14
        third_friday_date = datetime(target_year, target_month, third_friday)

    return third_friday_date.strftime('%Y-%m-%d')

def get_constituents():
    sp500 = 'https://yfiua.github.io/index-constituents/constituents-sp500.csv'

    sp500 = pd.read_csv(sp500)
    sp500 = sp500.sort_values("Symbol", ascending=True)
    sp500_constituents = sp500['Symbol'].to_list()
    sp500_constituents = [x.replace("-",".") for x in sp500_constituents]

    return sp500_constituents

def fetch_mid_iv(symbol, exp, spots):
    mid = spots.loc[symbol, 'mid_price']
    snap = get_chain_snapshot(symbol, mid, exp)
    calls = snap[snap['option_type']=="C"]
    puts = snap[snap['option_type']=="P"]
    calls = calls.reset_index(drop=True)
    puts = puts.reset_index(drop=True)
    closest_call = calls.iloc[(calls["strike_price"] - mid).abs().idxmin()]

    # Find the put with delta closest to -30
    closest_put = puts.iloc[(puts["strike_price"] - mid).abs().idxmin()]

    call_iv = closest_call['implied_volatility']
    put_iv = closest_put['implied_volatility']
    mid_iv = (put_iv+call_iv)/2
    print(mid_iv)
    exit()
    return mid_iv

def fetch_spx_iv(exp):
    request = StockLatestQuoteRequest(symbol_or_symbols="SPY")
    data = stock_historical_data_client.get_stock_latest_quote(request)
    ask = data["SPY"].ask_price
    bid = data["SPY"].bid_price
    mid = (bid+ask)/2
    snap = get_chain_snapshot("SPY", mid, exp)
    calls = snap[snap['option_type']=="C"]
    puts = snap[snap['option_type']=="P"]
    calls = calls.reset_index(drop=True)
    puts = puts.reset_index(drop=True)
    closest_call = calls.iloc[(calls["strike_price"] - mid).abs().idxmin()]

    # Find the put with delta closest to -30
    closest_put = puts.iloc[(puts["strike_price"] - mid).abs().idxmin()]

    call_iv = closest_call['implied_volatility']
    put_iv = closest_put['implied_volatility']
    mid_iv = (put_iv+call_iv)/2
    return mid_iv

def get_atm_constituent_options(sp, exp):
    request = StockLatestQuoteRequest(symbol_or_symbols=sp)
    data = stock_historical_data_client.get_stock_latest_quote(request)

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.applymap(lambda x: x[1] if isinstance(x, tuple) else x)
    df = df.reset_index(drop=True)
    df.columns = ['Symbol','timestamp', 'ask_price', 'ask_size', 'bid_exchange', 'bid_price', 'bid_size', 'conditions', 'tape', 'symbol']
    df = df.drop(columns=['timestamp', 'ask_size', 'bid_exchange', 'bid_size', 'conditions', 'tape', 'symbol'])
    df['mid_price'] = df.apply(
    lambda row: (row['ask_price'] + row['bid_price']) / 2 if row['ask_price'] != 0 and row['bid_price'] != 0 else 
                row['ask_price'] if row['bid_price'] == 0 else
                row['bid_price'], axis=1
    )
    spots = df[df['mid_price']>0].set_index('Symbol')

    for symbol in sp:
        try:
            mid_iv = fetch_mid_iv(symbol, exp, spots)
            print(mid_iv)
            exit()
        except KeyError:
            print("could not get snapshot for {}".format(symbol))

if __name__=="__main__":
    sp = get_constituents()
    exp = get_next_third_friday()
    # spx_iv = fetch_spx_iv(exp)

    implied_vols = get_atm_constituent_options(sp[:5], exp)

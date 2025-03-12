import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import time
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context
max_requests_per_minute = 150
delay_per_request = 60/max_requests_per_minute
load_dotenv()

ALPACA_API_KEY="PKSWP8A9QH87I5DX69Y4"
ALPACA_API_SECRET="we0Fo1a7UmijE45Z3rWYkZjXqnsfFg3B8pwrnbdC"
ALPACA_BASE_URL="https://paper-api.alpaca.markets/"

from alpaca.data.requests import OptionLatestQuoteRequest, OptionSnapshotRequest, OptionChainRequest
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
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

from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
skew_csv_path = os.path.join(script_dir, "dispersion.csv")
plot_dir = os.path.join(script_dir, "plots")
plot_path = os.path.join(plot_dir, "dispersion_plot.png")
implied_vols_path = os.path.join(script_dir, "implied_vols.pkl")
market_caps_dir = os.path.join(script_dir, "market_caps")
market_caps_path = os.path.join(market_caps_dir, "market_caps.xlsx")

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
        verify=False
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
    sp500_constituents = [x for x in sp500_constituents if x != "BF.B"]

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
    mid_iv = np.nanmean([put_iv, call_iv])
    return mid_iv

def fetch_spx_iv(exp):
    request = StockLatestQuoteRequest(symbol_or_symbols="SPY", verify=False)
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
    request = StockLatestQuoteRequest(symbol_or_symbols=sp,verify=False)
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

    mids, symbols = [], []

    for i, symbol in enumerate(sp, start=1):  # Start index at 1 for easier mod checks
        try:
            mid_iv = fetch_mid_iv(symbol, exp, spots)
            mids.append(mid_iv)
            symbols.append(symbol)
        except Exception as e:
            print("Error for {}: {}".format(symbol, e))
            continue
        
        if i % max_requests_per_minute == 0:  # If reaching limit, pause
            print("Reached 150 API calls, sleeping for 60 seconds...")
            time.sleep(60)
        else:
            time.sleep(delay_per_request)

    # for symbol in sp:
    #     try:
    #         mid_iv = fetch_mid_iv(symbol, exp, spots)
    #         mids.append(mid_iv)
    #         symbols.append(symbol)
    #     except KeyError:
    #         print("could not get mid_iv for {}".format(symbol))
    #         continue
    
    implied_vols = pd.DataFrame({
        "Symbol":symbols, 
        "IV": mids
    })

    return implied_vols

def plot():
    disp = pd.read_csv(skew_csv_path)
    disp.columns = ['Date', 'SPX ATM IV', 'Constit. ATM IV']
    disp['Dispersion'] = disp['SPX ATM IV']-disp['Constit. ATM IV']
    disp[['Dispersion', 'SPX ATM IV', 'Constit. ATM IV']] *= 100
    disp['RSI'] = calculate_rsi(disp['Dispersion'])
    disp['MACD'], disp['Signal Line'] = calculate_macd(disp['Dispersion'])

    disp['Long Dispersion'] = (disp['RSI'] < 30) & (disp['MACD'] > disp['Signal Line'])  # Buy signal (RSI < 30 and MACD > Signal Line)
    disp['Short Dispersion'] = (disp['RSI'] > 70) & (disp['MACD'] < disp['Signal Line'])    
    df = disp
    df['Date'] = pd.to_datetime(df['Date'])

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 15), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1, 1]})

    # Plot IVs and Dispersion on primary y-axis for top chart
    ax1.plot(disp["Date"], disp["SPX ATM IV"], label="SPX ATM IV", color='blue')
    ax1.plot(disp["Date"], disp["Constit. ATM IV"], label="Constit. ATM IV", color='green')
    ax1.set_ylabel("IV Values", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc="upper left")
    ax1.grid(True)
    ax1.set_title("Dispersion and IV over Time")

    # Create secondary y-axis for Dispersion
    ax2_1 = ax1.twinx()  # Secondary y-axis for Dispersion
    ax2_1.plot(disp["Date"], disp["Dispersion"], label="Dispersion", color='red', linestyle='dashed')
    ax2_1.set_ylabel("Dispersion", color='red')
    ax2_1.tick_params(axis='y', labelcolor='red')

    long_disp_dates = disp[disp['Long Dispersion'] == True]['Date']
    for date in long_disp_dates:
        ax2_1.scatter(date, disp[disp['Date'] == date]['Dispersion'].values[0], color='green', marker='^', s=100, label="Long Dispersion", zorder=5)

    # Short Dispersion - Red Downward Arrow
    short_disp_dates = disp[disp['Short Dispersion'] == True]['Date']
    for date in short_disp_dates:
        ax2_1.scatter(date, disp[disp['Date'] == date]['Dispersion'].values[0], color='red', marker='v', s=100, label="Short Dispersion", zorder=5)

    # Plot MACD and Signal Line on the middle chart
    ax2.plot(disp["Date"], disp["MACD"], label="MACD", color='purple', alpha=0.6)  # Dimmed MACD line
    ax2.plot(disp["Date"], disp["Signal Line"], label="Signal Line", color='orange', alpha=0.6)  # Dimmed Signal Line
    ax2.set_ylabel("MACD / Signal Line", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc="upper left")
    ax2.set_title("MACD and Signal Line")

    # Plot RSI on the bottom chart
    ax3.plot(disp["Date"], disp["RSI"], label="RSI", color='blue', linestyle='-', markersize=5)
    ax3.axhline(y=70, color='red', linestyle='--', label="Overbought (70)")  # Overbought line
    ax3.axhline(y=30, color='green', linestyle='--', label="Oversold (30)")  # Oversold line
    ax3.set_ylabel("RSI", color='black')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.legend(loc="upper left")
    ax3.set_title("Relative Strength Index (RSI)")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set overall figure title
    plt.suptitle("Dispersion, IV, RSI, and MACD over Time", fontsize=14)

    # Save plot
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Adjust space between the subplots
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return disp

def calculate_rsi(data, window=7):
    delta = data.diff()  # Calculate price change
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Average gain
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Average loss
    
    rs = gain / loss  # Relative strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi

def calculate_macd(series, fast=3, slow=8, signal=3):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def signal(window, implied_vols, long_signal, short_signal, sp):
    if long_signal or short_signal:
        end = datetime.today().date()
        start = end - timedelta(days=90)
        stock_bars_request = StockBarsRequest(symbol_or_symbols=sp+['SPY'], timeframe=TimeFrame.Day, start=start, end=end, verify=False)
        stock_bars = stock_historical_data_client.get_stock_bars(stock_bars_request)
        constituent_bars = stock_bars.df
        constituent_barst = constituent_bars.reset_index()

        # Pivot the dataframe to get the close prices as columns, indexed by timestamp
        constituent_bars = constituent_bars.pivot_table(index='timestamp', columns='symbol', values='close')
        constituent_bars.index = constituent_bars.index.date
        implied_vols = implied_vols.sort_values("IV")
        if long_signal: # we need high vol low corr const
            implied_vols = implied_vols.tail(len(implied_vols)//2)
            high_vol = [x for x in implied_vols.index]
            constituent_bars = constituent_bars[high_vol+["SPY"]]
            returns = constituent_bars.pct_change().dropna()
            rolling_corr = returns.rolling(window).corr(returns["SPY"]).dropna().tail(1)
            rolling_corr = rolling_corr.drop(columns="SPY")
            rolling_corr = rolling_corr.transpose()
            rolling_corr.columns=["Rolling Corr"]
            rolling_corr.columns.name = None
            rolling_corr = rolling_corr.sort_values("Rolling Corr", ascending=True)
            long_vega = rolling_corr.head(len(rolling_corr)//3)
            long_vega.index.name = "Symbol"
            long_vega = pd.merge(long_vega, implied_vols, left_index=True, right_index=True, how='inner')
            long_vega = long_vega.drop(columns=['Market Cap', 'Weight', 'Weight Adj IV'])
            return long_vega
        elif short_signal: # we need low vol high corr const
            implied_vols = implied_vols.head(len(implied_vols)//2)
            low_vol = [x for x in implied_vols.index]
            constituent_bars = constituent_bars[low_vol+["SPY"]]
            returns = constituent_bars.pct_change().dropna()
            rolling_corr = returns.rolling(window).corr(returns["SPY"]).dropna().tail(1)
            rolling_corr = rolling_corr.drop(columns="SPY")
            rolling_corr = rolling_corr.transpose()
            rolling_corr.columns=["Rolling Corr"]
            rolling_corr.columns.name = None
            rolling_corr = rolling_corr.sort_values("Rolling Corr", ascending=True)
            short_vega = rolling_corr.tail(len(rolling_corr)//3)
            short_vega.index.name = "Symbol"
            short_vega = pd.merge(short_vega, implied_vols, left_index=True, right_index=True, how='inner')
            short_vega = short_vega.drop(columns=['Market Cap', 'Weight', 'Weight Adj IV'])
            return short_vega
    else:
        return None

            
    
if __name__=="__main__":
    sp = get_constituents()
    exp = get_next_third_friday()
    spx_iv = fetch_spx_iv(exp)

    implied_vols = get_atm_constituent_options(sp, exp)
    implied_vols.to_pickle(implied_vols_path)
    implied_vols = pd.read_pickle(implied_vols_path)
    mkt_caps = pd.read_excel(market_caps_path, sheet_name='Market Caps')
    mkt_caps = mkt_caps[["Ticker", 'Market Cap']].rename(columns={'Ticker': 'Symbol'}).set_index('Symbol')
    implied_vols = implied_vols.set_index('Symbol')

    implied_vols = pd.merge(implied_vols, mkt_caps, left_index=True, right_index=True, how='inner')
    implied_vols['Weight'] = implied_vols['Market Cap']/implied_vols['Market Cap'].sum()
    implied_vols['Weight Adj IV'] = implied_vols['Weight'] * implied_vols['IV']
    weighted_sum_IV = implied_vols['Weight Adj IV'].sum()

    dispersion_df = pd.DataFrame({"Date":[datetime.today().date().strftime("%Y-%m-%d")], "SPX ATM IV":[spx_iv], "Constit. ATM IV":[weighted_sum_IV]})

    try:
        dispersion_df.to_csv(skew_csv_path, mode='a', header=False, index=False)
    except FileNotFoundError:
        dispersion_df.to_csv(skew_csv_path, mode='w', header=True, index=False)

    disp = plot()
    long_signal = disp.loc[len(disp)-1, "Long Dispersion"]
    short_signal = disp.loc[len(disp)-1, 'Short Dispersion']
    correlation_window = 30
    constituents_to_trade = signal(correlation_window, implied_vols, long_signal, short_signal, sp)
    
    if long_signal ==True:
        signal_string = "Short SPX Gamma, Long High Vol Low Corr Constit."

    elif short_signal == True:
        signal_string = "Long SPX Gamma, Short Low Vol High Corr Constit."
    else:
        signal_string = "No Action"

    # Email configuration
    SMTP_SERVER = "smtp.gmail.com"  # Gmail SMTP server
    SMTP_PORT = 587
    SENDER_EMAIL = "mmisic03@gmail.com"
    SENDER_PASSWORD = "ynbu lndn lfxx sulk"  # Use an app password if required
    RECIPIENT_EMAILS = ["casselrobson19@gmail.com", "misi2700@mylaurier.ca"]
    SUBJECT = f"Daily Dispersion - {signal_string} - {datetime.today().strftime('%Y-%m-%d')}"
    
    disp.iloc[:, 1:8] = disp.iloc[:, 1:8].round(2)
    ht_disp = disp.sort_values("Date", ascending=False).head(5).to_html(index=False, border=1, justify='center')
    if constituents_to_trade is not None:
        # Convert DataFrame to HTML table
        constituents_to_trade.index.name = None
        ht_constituents = constituents_to_trade.to_html(index=True, border=1, justify="center")

        # Create email message
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = ", ".join(RECIPIENT_EMAILS) 
        msg["Subject"] = SUBJECT

        # Email body
        email_body = f"""\
        <html>
        <head>
            <style>
                table {{border-collapse: collapse; width: 100%;}}
                th, td {{border: 1px solid black; padding: 8px; text-align: center;}}
                th {{background-color: #f2f2f2;}}
            </style>
        </head>
        <body>
            <h2>Daily Dispersion</h2>
            {ht_disp}
            <h2>Constituents to Trade</h2>
            {ht_constituents}
            <p>Plot of Dispersion Levels.</p>
        </body>
        </html>
        """
    else:
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = ", ".join(RECIPIENT_EMAILS) 
        msg["Subject"] = SUBJECT

        # Email body
        email_body = f"""\
        <html>
        <head>
            <style>
                table {{border-collapse: collapse; width: 100%;}}
                th, td {{border: 1px solid black; padding: 8px; text-align: center;}}
                th {{background-color: #f2f2f2;}}
            </style>
        </head>
        <body>
            <h2>Daily Dispersion</h2>
            {ht_disp}
            <p>Plot of Dispersion Levels.</p>
        </body>
        </html>
        """

    msg.attach(MIMEText(email_body, "html"))

    # Attach plot
    if os.path.exists(plot_path):
        with open(plot_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(plot_path)}")
            msg.attach(part)

    # Send the email
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
import matplotlib.pyplot as plt

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

warnings.filterwarnings('ignore')

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
    sp500_constituents = [x.replace(".","-") for x in sp500_constituents]

    return sp500_constituents

def fetch_spx_options_data():
    spx = yf.Ticker('^SPX')
    
    exp = get_next_third_friday()

    options_data = spx.option_chain(exp)  # Choose the nearest expiration

    # Combine the calls and puts into one DataFrame
    calls = options_data.calls
    puts = options_data.puts
    spx_price = spx.history(period='1d')['Close'][0]
    
    all_strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
    closest_strike = min(all_strikes, key=lambda x: abs(x - spx_price))
    

    # Filter options to only include the closest strike price
    calls = calls[calls['strike'] == closest_strike].reset_index(drop=True)
    puts = puts[puts['strike'] == closest_strike].reset_index(drop=True)

    c_iv = calls.loc[0, "impliedVolatility"]
    p_iv = puts.loc[0, 'impliedVolatility']
    mid_iv = (c_iv+p_iv)/2
            
    
    return mid_iv, exp

def get_ticker_spot(ticker):
    ticker_spot = ticker.history(period='1d')['Close'][0]
    mkcap = ticker.info.get('marketCap')
    return ticker_spot, mkcap

def get_rolling_iv(ticker):
    df = pd.read_csv('iv_historical.csv')
    if len(df) <=30:
        rolling_iv = df[ticker].rolling(window=len(df)-1).mean().tail(1).values[0]
    else:
        rolling_iv = df[ticker].rolling(window=30).mean().tail(1).values[0]
    return rolling_iv

def get_atm_constituent_options(sp, exp):
    # Batch download spot prices and market caps
    tickers_data = yf.download(sp, period="1d", progress=False)['Close']
    mids, symbols, caps = [], [], []
    
    for constituent in sp:
        print('here')
        try:
            spot = tickers_data[constituent] if constituent in tickers_data else None
            if spot is None or pd.isna(spot):
                continue  # Skip if no valid spot price

            constituent_ticker = yf.Ticker(constituent)
            options_data = constituent_ticker.option_chain(exp)
            
            all_strikes = sorted(set(options_data.calls['strike']).union(set(options_data.puts['strike'])))
            closest_strike = min(all_strikes, key=lambda x: abs(x - spot))

            # Find ATM option
            call_iv = options_data.calls[options_data.calls['strike'] == closest_strike]['impliedVolatility']
            put_iv = options_data.puts[options_data.puts['strike'] == closest_strike]['impliedVolatility']
            
            if call_iv.empty or put_iv.empty:
                continue  # Skip if no valid IV

            mid_iv = (call_iv.iloc[0] + put_iv.iloc[0]) / 2
            mids.append(mid_iv)
            symbols.append(constituent)
            caps.append(None)  # You can optimize market cap retrieval separately

        except Exception as e:
            print(f"Error fetching {constituent}: {e}")
            continue

    # Create DataFrame
    implied_vols = pd.DataFrame({
        "Symbol": symbols,
        "IV": mids,
        "Market Cap": caps
    })
    return implied_vols
# def get_atm_constituent_options(sp, exp):
#     mids = []
#     symbols = []
#     caps = []
#     for constituent in sp:
#         try:
#             constituent_ticker = yf.Ticker(constituent)
#             options_data = constituent_ticker.option_chain(exp)
#             calls = options_data.calls
#             puts = options_data.puts
#             spot, mkcap = get_ticker_spot(constituent_ticker)
#             calls = calls[(calls['strike']>spot-2.5)&(calls['strike']<spot+2.5)]
#             puts = puts[(puts['strike']>spot-2.5)&(puts['strike']<spot+2.5)]
#             calls = calls.reset_index().drop(columns='index')
#             puts = puts.reset_index().drop(columns='index')
#             c_iv = calls.loc[0, "impliedVolatility"]
#             p_iv = puts.loc[0, 'impliedVolatility']
#             mid_iv = (c_iv+p_iv)/2
#             mids.append(mid_iv)
#             symbols.append(constituent)
#             caps.append(mkcap)
#         except Exception as e:
#             print(e)
#             continue
#     implied_vols = pd.DataFrame({
#         "Symbol":symbols,
#         "IV":mids,
#         "Market Cap":caps
#     })
#     return implied_vols

def store_new_implied_vol(df, spx_iv):
    store_iv = df
    store_iv = store_iv[["Symbol", "IV"]]
    today = datetime.today().date()
    today = today.strftime("%Y-%m-%d")
    store_iv = store_iv.rename(columns={"IV":today})
    store_iv = store_iv.set_index("Symbol").transpose()
    store_iv = store_iv.reset_index().rename(columns={"index":"Date"})
    store_iv.columns.name = None
    store_iv["SPX"] = spx_iv

    csv_file = "iv_historical.csv"
    if os.path.exists(csv_file):
        store_iv.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        store_iv.to_csv(csv_file, mode='w', header=True, index=False)
    return

def get_rolling_ivs(sp, df):
    df['Rolling IV'] = None
    df = df.set_index("Symbol")
    for ticker in df.index:
        df.loc[ticker, "Rolling IV"] = get_rolling_iv(ticker)

    return df.reset_index()

def plot():
    plot_disp = pd.read_csv("dispersion.csv")
    plot_disp.columns = ['Date', 'SPX ATM IV', 'Constit. ATM IV']
    plot_disp['Dispersion'] = plot_disp['SPX ATM IV']-plot_disp["Constit. ATM IV"]
    df = plot_disp
    df['Date'] = pd.to_datetime(df['Date'])
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Call IV and Put IV on primary y-axis
    ax1.plot(df["Date"], df["SPX ATM IV"], label="SPX ATM IV", marker='o', color='blue')
    ax1.plot(df["Date"], df["Constit. ATM IV"], label="Constit. ATM IV", marker='s', color='green')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("IV Values", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Create secondary y-axis for Skew
    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["Dispersion"], label="Dispersion", marker='^', color='red', linestyle='dashed')
    ax2.set_ylabel("Dispersion", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc="upper right")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.title("Dispersion over Time")
    # Save plot
    plt.savefig(r"C:\Users\Cassel Robson\skewmonitor\plots\dispersion_plot.png", bbox_inches="tight")
    plt.close()
    return plot_disp

def calculate_z_score(series, lookback=5):
    mean = series.rolling(window=lookback).mean()
    std = series.rolling(window=lookback).std()
    return (series - mean) / std

def calculate_macd(series, fast=3, slow=8, signal=3):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_ohlc(symbols, start_date, end_date):
    data = {}
    for symbol in symbols:
        print(symbol)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        data[symbol] = df[['Close']]
    
    # Combine the data into a single DataFrame with multi-level columns
    combined_df = pd.concat(data, axis=1)
    return combined_df

def signal(window_size, df, long_signal, short_signal):
    end = datetime.today().date()
    start = end - timedelta(days=window_size+18)
    spx_prices = get_ohlc(["^SPX"], start, end)
    spx_prices.to_pickle('spx.pkl')
    spx_prices = pd.read_pickle('spx.pkl')
    #long we need lowest correlation
    #short we need highest correlation
    if long_signal:
        df = df[df['IV'] < df['Rolling IV']]
        universe = [x for x in df['Symbol']]
        universe = get_ohlc(universe, start, end)
        universe.to_pickle('universe.pkl')
        universe = pd.read_pickle('universe.pkl')
        spx_prices.index = pd.to_datetime(spx_prices.index)
        universe.index = pd.to_datetime(universe.index)
        spx_prices.columns = spx_prices.columns.droplevel(1)
        universe.columns = universe.columns.droplevel(1)

        # Compute returns
        spx_returns = spx_prices.pct_change().dropna()
        returns = universe.pct_change().dropna()

        # Ensure `spx_returns` is a DataFrame (not Series) and has the correct column name
        spx_returns = spx_returns.rename(columns={"^SPX": "SPX"})

        # Compute rolling correlation for each stock vs SPX
        window_size = 30
        rolling_corr = returns.rolling(window_size).corr(spx_returns["SPX"]).dropna().tail(1)
        rolling_corr = rolling_corr.transpose()
        rolling_corr.columns = ["Rolling Corr"]
        rolling_corr.columns.name = None
        rolling_corr = rolling_corr.sort_values("Rolling Corr", ascending=True)
        long_vol = rolling_corr.head(len(rolling_corr)//3)
        df = df.set_index('Symbol')
        df = df[df.index.isin(long_vol.index)]
        long_vol = pd.concat([df, long_vol], axis=1)
        return long_vol
    elif short_signal:
        df = df[df['IV'] > df['Rolling IV']]
        universe = [x for x in df['Symbol']]
        universe = get_ohlc(universe, start, end)
        universe.to_pickle('universe.pkl')
        universe = pd.read_pickle('universe.pkl')
        spx_prices.index = pd.to_datetime(spx_prices.index)
        universe.index = pd.to_datetime(universe.index)
        spx_prices.columns = spx_prices.columns.droplevel(1)
        universe.columns = universe.columns.droplevel(1)

        # Compute returns
        spx_returns = spx_prices.pct_change().dropna()
        returns = universe.pct_change().dropna()

        # Ensure `spx_returns` is a DataFrame (not Series) and has the correct column name
        spx_returns = spx_returns.rename(columns={"^SPX": "SPX"})

        # Compute rolling correlation for each stock vs SPX
        window_size = 30
        rolling_corr = returns.rolling(window_size).corr(spx_returns["SPX"]).dropna().tail(1)
        rolling_corr = rolling_corr.transpose()
        rolling_corr.columns = ["Rolling Corr"]
        rolling_corr.columns.name = None
        rolling_corr = rolling_corr.sort_values("Rolling Corr", ascending=True)
        short_vol = rolling_corr.tail(len(rolling_corr)//3)
        df = df.set_index('Symbol')
        df = df[df.index.isin(short_vol.index)]
        short_vol = pd.concat([df, short_vol], axis=1)
        return short_vol
    else: return None

if __name__ == "__main__":
    sp = get_constituents()
    spx_iv, exp = fetch_spx_options_data()
    df = get_atm_constituent_options(sp, exp)
    df.to_pickle('implieds.pkl')
    df = pd.read_pickle('implieds.pkl')
    store_new_implied_vol(df, spx_iv)
    df = get_rolling_ivs(sp, df)
    df = df.sort_values('Market Cap', ascending=False)
    df['Weight'] = df['Market Cap']/df['Market Cap'].sum()
    df['Weight Adj IV'] = df['Weight']*df['IV']
    weighted_sum_iv = df['Weight Adj IV'].sum()
    

    dispersion_gap = spx_iv - weighted_sum_iv
    
    dispersion_df = pd.DataFrame({"Date":[datetime.today().date().strftime("%Y-%m-%d")], "SPX ATM IV":[spx_iv], "Constit. ATM IV":[weighted_sum_iv]})

    try:
        dispersion_df.to_csv("dispersion.csv", mode='a', header=False, index=False)
    except FileNotFoundError:
        dispersion_df.to_csv('dispersion.csv', mode='w', header=True, index=False)
    
    disp = plot()

    disp['Z-Score'] = calculate_z_score(disp['Dispersion'])
    disp['MACD'], disp['Signal Line'] = calculate_macd(disp['Dispersion'])

    # Trading Signals
    disp['Long Signal'] = (disp['Z-Score'] < -2) & (disp['MACD'] > disp['Signal Line'])
    disp['Short Signal'] = (disp['Z-Score'] > 2) & (disp['MACD'] < disp['Signal Line'])
    long_signal = disp.loc[len(disp)-1, "Long Signal"]
    short_signal = disp.loc[len(disp)-1, 'Short Signal']
    window_size = 30
    df.to_pickle('df.pkl')
    df = pd.read_pickle('df.pkl')
    
    
    long_signal = True
    short_signal = False


    constituents_to_trade = signal(window_size, df, long_signal, short_signal)

    if long_signal ==True:
        signal_string = "Short SPX Gamma, Long Constit."

    elif short_signal == True:
        signal_string = "Long SPX Gamma, Short Constit."
    else:
        signal_string = "No Action"

    

    plot_path = r"C:\Users\Cassel Robson\skewmonitor\plots\dispersion_plot.png"
    # Email configuration
    SMTP_SERVER = "smtp.gmail.com"  # Change based on your email provider
    SMTP_PORT = 587
    SENDER_EMAIL = "casselrobson93@gmail.com"
    SENDER_PASSWORD = "lajhhtqevwvomcts"  # Use an app password if using Gmail
    RECIPIENT_EMAIL = "casselrobson19@gmail.com"
    SUBJECT = f"Daily Dispersion - {signal_string} - {datetime.today().strftime('%Y-%m-%d')}"


    ht_disp = disp.to_html(index=False, border=1, justify='center')
    if long_signal == True:
        # Convert DataFrame to HTML table
        ht_constituents = constituents_to_trade.to_html(index=True, border=1, justify="center")

        # Create email message
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
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
    elif short_signal == True:
        # Convert DataFrame to HTML table
        ht_constituents = constituents_to_trade.to_html(index=True, border=1, justify="center")

        # Create email message
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
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
        # Convert DataFrame to HTML table
        ht_constituents = constituents_to_trade.to_html(index=True, border=1, justify="center")

        # Create email message
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
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
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

    

    
import yfinance as yf
import pandas as pd
from datetime import datetime
import warnings
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
def get_constituents():
    sp500 = 'https://yfiua.github.io/index-constituents/constituents-sp500.csv'

    sp500 = pd.read_csv(sp500)
    sp500 = sp500.sort_values("Symbol", ascending=True)
    sp500_constituents = sp500['Symbol'].to_list()
    sp500_constituents = [x.replace(".","-") for x in sp500_constituents]

    return sp500_constituents

def fetch_spx_options_data():
    spx = yf.Ticker('^SPX')
    
    # Get options expiration dates for SPX
    expiration_dates = spx.options
    # Fetch the options data for the nearest expiration date (or specify one)

    exp = expiration_dates[22]

    options_data = spx.option_chain(exp)  # Choose the nearest expiration

    # Combine the calls and puts into one DataFrame
    calls = options_data.calls

    puts = options_data.puts
    spx_price = spx.history(period='1d')['Close'][0]
    calls = calls[(calls['strike']>spx_price-2.5)&(calls['strike']<spx_price+2.5)]
    puts = puts[(puts['strike']>spx_price-2.5)&(puts['strike']<spx_price+2.5)]
    calls = calls.reset_index().drop(columns='index')
    puts = puts.reset_index().drop(columns='index')
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
    mids = []
    symbols = []
    caps = []
    for constituent in sp[:5]:
        try:
            constituent_ticker = yf.Ticker(constituent)
            options_data = constituent_ticker.option_chain(exp)
            calls = options_data.calls
            puts = options_data.puts
            spot, mkcap = get_ticker_spot(constituent_ticker)
            calls = calls[(calls['strike']>spot-2.5)&(calls['strike']<spot+2.5)]
            puts = puts[(puts['strike']>spot-2.5)&(puts['strike']<spot+2.5)]
            calls = calls.reset_index().drop(columns='index')
            puts = puts.reset_index().drop(columns='index')
            c_iv = calls.loc[0, "impliedVolatility"]
            p_iv = puts.loc[0, 'impliedVolatility']
            mid_iv = (c_iv+p_iv)/2
            mids.append(mid_iv)
            symbols.append(constituent)
            caps.append(mkcap)
        except Exception as e:
            print(e)
            continue
    implied_vols = pd.DataFrame({
        "Symbol":symbols,
        "IV":mids,
        "Market Cap":caps
    })
    return implied_vols

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
    plt.savefig("plots/dispersion_plot.png", bbox_inches="tight")
    plt.close()
    return plot_disp

def calculate_z_score(series, lookback=20):
    mean = series.rolling(window=lookback).mean()
    std = series.rolling(window=lookback).std()
    return (series - mean) / std

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


if __name__ == "__main__":
    # sp = get_constituents()
    # spx_iv, exp = fetch_spx_options_data()
    # # df = get_atm_constituent_options(sp, exp)
    # # df.to_pickle('implieds.pkl')
    # df = pd.read_pickle('implieds.pkl')
    # store_new_implied_vol(df, spx_iv)
    # df = get_rolling_ivs(sp, df)
    # df = df.sort_values('Market Cap', ascending=False)
    # df['Weight'] = df['Market Cap']/df['Market Cap'].sum()
    # df['Weight Adj IV'] = df['Weight']*df['IV']
    # weighted_sum_iv = df['Weight Adj IV'].sum()
    

    # dispersion_gap = spx_iv - weighted_sum_iv
    
    # dispersion_df = pd.DataFrame({"Date":[datetime.today().date().strftime("%Y-%m-%d")], "SPX ATM IV":[spx_iv], "Constit. ATM IV":[weighted_sum_iv]})

    # try:
    #     dispersion_df.to_csv("dispersion.csv", mode='a', header=False, index=False)
    # except FileNotFoundError:
    #     dispersion_df.to_csv('dispersion.csv', mode='w', header=True, index=False)
    
    disp = plot()
    df = disp
    df['Z-Score'] = calculate_z_score(df['Dispersion'])
    df['MACD'], df['Signal Line'] = calculate_macd(df['Dispersion'])

    # Trading Signals
    df['Long Entry'] = (df['Z-Score'] < -2) & (df['MACD'] > df['Signal Line'])
    df['Short Entry'] = (df['Z-Score'] > 2) & (df['MACD'] < df['Signal Line'])
    
    # if dispersion gap is positive, short index IV (short SPX straddle)
    # Long SSO with lowest relative IV (compare to 30d average)

    # if dispersion gap is negative, long index vol, and short high IV constituents. 
    


    
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
from email.mime.base import MIMEBase
from email import encoders
import warnings
import numpy as np
import re
warnings.filterwarnings('ignore')
load_dotenv()

ALPACA_API_KEY="PKSWP8A9QH87I5DX69Y4"
ALPACA_API_SECRET="we0Fo1a7UmijE45Z3rWYkZjXqnsfFg3B8pwrnbdC"
ALPACA_BASE_URL="https://paper-api.alpaca.markets/"

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


script_dir = os.path.dirname(os.path.abspath(__file__))
market_caps_dir = os.path.join(script_dir, "market_caps")
market_caps_path = os.path.join(market_caps_dir, "market_caps.xlsx")
plots_dir = os.path.join(script_dir, "plots")

stock_historical_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

max_requests_per_minute = 150
delay_per_request = 60/max_requests_per_minute

def get_constituents(): 
    sp500 = 'https://yfiua.github.io/index-constituents/constituents-sp500.csv'

    sp500 = pd.read_csv(sp500)
    sp500 = sp500.sort_values("Symbol", ascending=True)
    sp500_constituents = sp500['Symbol'].to_list()
    sp500_constituents = [x.replace("-",".") for x in sp500_constituents]
    sp500_constituents = [x for x in sp500_constituents if x != "BF.B"]

    return sp500_constituents

def retrieve_daily_close_prices_for_constituents(constituents):
    """
    Retrieve daily close prices for a list of stock constituents.

    Args:
        constituents: List of stock symbols

    Returns:
        DataFrame with dates as index, tickers as columns, close prices as values
    """
    data = {}
    error_symbols = []

    # Calculate date range (e.g., last 252 trading days ~ 1 year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    for i, symbol in enumerate(constituents, start=1):
        try:
            # Request daily bars for the symbol
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )

            bars = stock_historical_data_client.get_stock_bars(request_params)
            print(bars)
            exit()
            # Convert to dataframe and extract close prices
            df_bars = bars.df
            if not df_bars.empty:
                # Reset index to get timestamp as a column, then set it as index
                df_bars = df_bars.reset_index(level=0, drop=True)  # Drop symbol level, keep timestamp
                close_prices = df_bars['close']
                data[symbol] = close_prices
            else:
                print(f"No data returned for {symbol}")
                error_symbols.append(symbol)

        except Exception as e:
            print("Error for {}: {}".format(symbol, e))
            error_symbols.append(symbol)
            continue

        # Rate limiting: sleep after every max_requests_per_minute calls
        if i % max_requests_per_minute == 0:
            print("Reached 150 API calls, sleeping for 60 seconds")
            time.sleep(60)

    # Combine all symbols into a single DataFrame
    df = pd.DataFrame(data)

    # Print summary
    print(f"\nSuccessfully retrieved data for {len(data)} symbols")
    print(f"Failed symbols ({len(error_symbols)}): {error_symbols}")

    return df

def calculate_momentum_factor(df, lookback_period=365, exclude_recent_days=30):
    df = df.copy()
    df = df.pct_change()
    # Get the most recent date
    end_date = df.index[-1]
    
    # Calculate the start and end of the momentum period
    # End of momentum period = most recent date - exclude_recent_days
    end_of_momentum_period = end_date - pd.Timedelta(days=exclude_recent_days)
    start_of_momentum_period = end_date - pd.Timedelta(days=lookback_period)

    # Get prices at start and end of momentum period
    df = df.loc[(df.index >= start_of_momentum_period) & (df.index <= end_of_momentum_period)]
    df = df.cumsum()
    df = df.loc[df.index == end_of_momentum_period]
    df = df.T
    df.columns = ['momentum today']
    df = df.sort_values(by='momentum today', ascending=False)

    return df

def plot_and_save(df, title, path, figsize=(10,6), dpi=300):
    ax = df.plot(title=title, figsize=figsize)
    fig = ax.get_figure()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

from fractions import Fraction

def adjust_prices_for_splits_inferred(px: pd.DataFrame, max_den=10, tol=0.01, min_jump=0.20):
    """
    Infer and adjust stock splits from raw close prices (no split dates needed).

    px: wide DF of prices, DatetimeIndex, columns = tickers
    max_den: allow split ratios up to p/q where max(p,q) <= max_den (captures 2-for-1, 3-for-2, etc.)
    tol: relative tolerance for matching ratio (1% default)
    min_jump: require a big 1-day move (abs return >= 20%) to consider as split candidate
    """
    px = px.sort_index().astype(float)
    out = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)

    for col in px.columns:
        s = px[col].dropna()
        if len(s) < 3:
            out[col] = px[col]
            continue

        # ratio between consecutive closes (prev / cur)
        ratio = s.shift(1) / s
        ret = s.pct_change()

        # approximate ratio by a simple fraction p/q with small denominator
        cand = ratio.apply(lambda x: float(Fraction(x).limit_denominator(max_den)) if pd.notna(x) else np.nan)

        # split candidates: big move AND ratio close to a simple fraction AND not ~1
        close_match = (ratio - cand).abs() / ratio.abs() <= tol
        big_move = ret.abs() >= min_jump
        not_one = (cand - 1.0).abs() >= 0.05  # ignore tiny adjustments

        split_mask = close_match & big_move & not_one

        # multiplier that should apply to the *previous* day and all earlier history
        # if ratio≈k (e.g., 2-for-1 => ratio≈2), scale history by 1/k
        hist_mult = (1.0 / cand).where(split_mask).shift(-1).fillna(1.0)

        # cumulative factor applied to each date (affects all prior dates, not post-split)
        adj_factor = hist_mult[::-1].cumprod()[::-1]

        out[col] = (s * adj_factor).reindex(px.index)

    return out

if __name__ == "__main__":
    constituents = get_constituents()

    df = retrieve_daily_close_prices_for_constituents(constituents)

    df = adjust_prices_for_splits_inferred(df, min_jump=0.12, max_den=20, tol=0.015)

    prices = df.copy()

    df = calculate_momentum_factor(df, lookback_period=365, exclude_recent_days=30)

    
    mkt_caps = pd.read_excel(market_caps_path, sheet_name='Market Caps')
    mkt_caps = mkt_caps[["Ticker", 'Market Cap']].rename(columns={'Ticker': 'Symbol'}).set_index('Symbol')
    
    final_df = df.join(mkt_caps, how='left')
    final_df['momentum today'] = final_df['momentum today'] * 100
    final_df = final_df.rename(columns={'momentum today': '12-1M Momentum'})

    def format_market_cap(x):
        if pd.isna(x):
            return np.nan
        x = float(x)
        for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
            if abs(x) >= div:
                return f"{x/div:,.2f}{unit}"
        return f"{x:,.0f}"

    final_df["Market Cap (fmt)"] = final_df["Market Cap"].map(format_market_cap)

    # Sort still uses the numeric column


    pos_momentum = final_df.head(10).sort_values("Market Cap", ascending=False).drop(columns=["Market Cap"]).rename(columns={"Market Cap (fmt)": "Market Cap"})
    neg_momentum = final_df.tail(10).sort_values("Market Cap", ascending=False).drop(columns=["Market Cap"]).rename(columns={"Market Cap (fmt)": "Market Cap"})
    
    pos_momentum["12-1M Momentum"] = pos_momentum["12-1M Momentum"].map(
    lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
)
    neg_momentum["12-1M Momentum"] = neg_momentum["12-1M Momentum"].map(
    lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
)

    winners = [x for x in pos_momentum.index]
    losers = [x for x in neg_momentum.index]
    winners_prices = prices[winners].pct_change().cumsum()
    losers_prices = prices[losers].pct_change().cumsum()



    plot_and_save(winners_prices, "Top 10 Momentum Stocks Performance",
              os.path.join(plots_dir, "winners.png"))

    plot_and_save(losers_prices, "Bottom 10 Momentum Stocks Performance",
                os.path.join(plots_dir, "losers.png"))
    

    # email writing
    SMTP_SERVER = "smtp.gmail.com"  # Gmail SMTP server
    SMTP_PORT = 587
    SENDER_EMAIL = "casselrobson93@gmail.com"
    SENDER_PASSWORD = "lajhhtqevwvomcts"  
    RECIPIENT_EMAILS = ["casselrobson19@gmail.com", "misi2700@mylaurier.ca"]
    SUBJECT = f"Daily Momentum LS Report - {datetime.today().strftime('%Y-%m-%d')}"

    # Convert DataFrame to HTML table

    winners_html = pos_momentum.to_html(index=True, border=1, justify="center")
    losers_html = neg_momentum.to_html(index=True, border=1, justify="center")
    # Create email message
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECIPIENT_EMAILS)
    msg["Subject"] = SUBJECT

    # Email body
    email_body = f"""
    <html>
    <head>
        <style>
            table {{border-collapse: collapse; width: 100%;}}
            th, td {{border: 1px solid black; padding: 8px; text-align: center;}}
            th {{background-color: #f2f2f2;}}
        </style>
    </head>
    <body>
        <h2>Constituents with highest 12-1M momentum</h2>
        {winners_html}
        <h2>Constituents with lowest 12-1M momentum</h2>
        {losers_html}
    </body>
    </html>
    """

    msg.attach(MIMEText(email_body, "html"))

    # Attach plot
    if os.path.exists(plots_dir):
        with open(os.path.join(plots_dir, "winners.png"), "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename=winners.png")
            msg.attach(part)
        with open(os.path.join(plots_dir, "losers.png"), "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename=losers.png")
            msg.attach(part)
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
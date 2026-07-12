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
from fractions import Fraction
warnings.filterwarnings('ignore')
load_dotenv()

ALPACA_API_KEY = "PKSWP8A9QH87I5DX69Y4"
ALPACA_API_SECRET = "we0Fo1a7UmijE45Z3rWYkZjXqnsfFg3B8pwrnbdC"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/"

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

script_dir = os.path.dirname(os.path.abspath(__file__))
market_caps_dir = os.path.join(script_dir, "market_caps")
market_caps_path = os.path.join(market_caps_dir, "market_caps.xlsx")
plots_dir = os.path.join(script_dir, "plots")

stock_historical_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

max_requests_per_minute = 150
delay_per_request = 60 / max_requests_per_minute


def get_constituents():
    sp500 = 'https://yfiua.github.io/index-constituents/constituents-sp500.csv'
    sp500 = pd.read_csv(sp500)
    sp500 = sp500.sort_values("Symbol", ascending=True)
    sp500_constituents = sp500['Symbol'].to_list()
    sp500_constituents = [x.replace("-", ".") for x in sp500_constituents]
    sp500_constituents = [x for x in sp500_constituents if x != "BF.B"]
    return sp500_constituents


def retrieve_daily_close_prices_for_constituents(constituents):
    data = {}
    error_symbols = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)

    for i, symbol in enumerate(constituents, start=1):
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            bars = stock_historical_data_client.get_stock_bars(request_params)
            df_bars = bars.df
            if not df_bars.empty:
                df_bars = df_bars.reset_index(level=0, drop=True)
                data[symbol] = df_bars['close']
            else:
                print(f"No data returned for {symbol}")
                error_symbols.append(symbol)
        except Exception as e:
            print("Error for {}: {}".format(symbol, e))
            error_symbols.append(symbol)
            continue

        if i % max_requests_per_minute == 0:
            print("Reached 150 API calls, sleeping for 60 seconds")
            time.sleep(60)

    df = pd.DataFrame(data)
    print(f"\nSuccessfully retrieved data for {len(data)} symbols")
    print(f"Failed symbols ({len(error_symbols)}): {error_symbols}")
    return df


def adjust_prices_for_splits_inferred(px: pd.DataFrame, max_den=10, tol=0.01, min_jump=0.20):
    px = px.sort_index().astype(float)
    out = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)

    for col in px.columns:
        s = px[col].dropna()
        if len(s) < 3:
            out[col] = px[col]
            continue

        ratio = s.shift(1) / s
        ret = s.pct_change()
        cand = ratio.apply(lambda x: float(Fraction(x).limit_denominator(max_den)) if pd.notna(x) else np.nan)

        close_match = (ratio - cand).abs() / ratio.abs() <= tol
        big_move = ret.abs() >= min_jump
        not_one = (cand - 1.0).abs() >= 0.05

        split_mask = close_match & big_move & not_one
        hist_mult = (1.0 / cand).where(split_mask).shift(-1).fillna(1.0)
        adj_factor = hist_mult[::-1].cumprod()[::-1]
        out[col] = (s * adj_factor).reindex(px.index)

    return out


def calculate_momentum_factor(df, lookback_period=90, exclude_recent_days=7):
    df = df.copy()
    df = df.pct_change()
    end_date = df.index[-1]
    end_of_momentum_period = end_date - pd.Timedelta(days=exclude_recent_days)
    start_of_momentum_period = end_date - pd.Timedelta(days=lookback_period)
    df = df.loc[(df.index >= start_of_momentum_period) & (df.index <= end_of_momentum_period)]
    df = df.cumsum()
    df = df.loc[df.index == end_of_momentum_period]
    df = df.T
    df.columns = ['momentum']
    df = df.sort_values(by='momentum', ascending=False)
    return df


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Compute MACD line, signal line, and histogram for all tickers."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def screen_macd_reversals(prices, momentum_df, momentum_universe=75, crossover_window=7, mode='reversal'):
    """
    mode='reversal'     — strong momentum stocks where MACD is turning against the trend
    mode='continuation' — strong momentum stocks where MACD pulled back but is resuming the trend
    """
    macd_line, signal_line, histogram = calculate_macd(prices)

    winners = momentum_df.head(momentum_universe).index.tolist()
    losers = momentum_df.tail(momentum_universe).index.tolist()

    majority = (crossover_window // 2) + 1

    def classify(ticker, look_for_bearish):
        if ticker not in histogram.columns:
            return None
        hist = histogram[ticker].dropna()
        if len(hist) < crossover_window + 2:
            return None

        recent = hist.iloc[-crossover_window:]
        signs = np.sign(hist)
        sign_change = signs.diff()
        recent_changes = sign_change.iloc[-crossover_window:]

        current_hist = hist.iloc[-1]
        current_macd = macd_line[ticker].dropna().iloc[-1] if ticker in macd_line.columns else np.nan
        current_signal = signal_line[ticker].dropna().iloc[-1] if ticker in signal_line.columns else np.nan
        mom_pct = momentum_df.loc[ticker, 'momentum'] * 100 if ticker in momentum_df.index else np.nan

        def base_row(signal):
            return {
                'Signal': signal,
                'Days Since Cross': None,
                'MACD Histogram': round(current_hist, 4),
                'MACD Line': round(current_macd, 4),
                'Signal Line': round(current_signal, 4),
                '3M-1W Momentum': f"{mom_pct:.2f}%",
            }

        if look_for_bearish:
            if (recent_changes == -1).any():
                cross_idx = recent_changes[recent_changes == -1].index[-1]
                days_ago = (hist.index[-1] - cross_idx).days
                row = base_row('Bearish Crossover')
                row['Days Since Cross'] = days_ago
                return row
            hist_diffs = recent.diff().dropna()
            declining_days = (hist_diffs < 0).sum()
            if recent.iloc[-1] > 0 and declining_days >= majority:
                return base_row('Decelerating (Approaching Bearish)')
            if recent.iloc[-1] > 0 and current_hist < recent.iloc[0]:
                return base_row('Weakening')

        else:
            if (recent_changes == 1).any():
                cross_idx = recent_changes[recent_changes == 1].index[-1]
                days_ago = (hist.index[-1] - cross_idx).days
                row = base_row('Bullish Crossover')
                row['Days Since Cross'] = days_ago
                return row
            hist_diffs = recent.diff().dropna()
            rising_days = (hist_diffs > 0).sum()
            if recent.iloc[-1] < 0 and rising_days >= majority:
                return base_row('Decelerating (Approaching Bullish)')
            if recent.iloc[-1] < 0 and current_hist > recent.iloc[0]:
                return base_row('Weakening')

        return None

    # Determine what signal direction to look for based on group + mode
    # Reversal:     winners -> bearish,  losers -> bullish
    # Continuation: winners -> bullish,  losers -> bearish
    winners_look_bearish = (mode == 'reversal')
    losers_look_bearish  = (mode == 'continuation')

    group_a_rows = {}
    for ticker in winners:
        result = classify(ticker, look_for_bearish=winners_look_bearish)
        if result:
            group_a_rows[ticker] = result

    group_b_rows = {}
    for ticker in losers:
        result = classify(ticker, look_for_bearish=losers_look_bearish)
        if result:
            group_b_rows[ticker] = result

    group_a_df = pd.DataFrame(group_a_rows).T if group_a_rows else pd.DataFrame()
    group_b_df = pd.DataFrame(group_b_rows).T if group_b_rows else pd.DataFrame()

    def sort_key(df):
        if df.empty:
            return df
        df = df.copy()
        signal_order = {
            'Bearish Crossover': 0, 'Bullish Crossover': 0,
            'Decelerating (Approaching Bearish)': 1, 'Decelerating (Approaching Bullish)': 1,
            'Weakening': 2,
        }
        df['_sort_tier'] = df['Signal'].map(signal_order).fillna(3)
        df['_sort_hist'] = df['MACD Histogram'].abs().astype(float)
        df = df.sort_values(['_sort_tier', '_sort_hist'], ascending=[True, True])
        return df.drop(columns=['_sort_tier', '_sort_hist'])

    return {
        'group_a': sort_key(group_a_df),  # top momentum stocks
        'group_b': sort_key(group_b_df),  # bottom momentum stocks
    }


def plot_macd(prices, ticker, macd_line, signal_line, histogram, ax1, ax2):
    """Plot price and MACD panel for a single ticker onto provided axes."""
    ax1.plot(prices.index, prices[ticker], linewidth=1.5)
    ax1.set_title(f"{ticker} — Price & MACD")
    ax1.set_ylabel("Price")

    hist_vals = histogram[ticker].dropna()
    colors = ['green' if v >= 0 else 'red' for v in hist_vals]
    ax2.bar(hist_vals.index, hist_vals.values, color=colors, alpha=0.6, label='Histogram')
    ax2.plot(macd_line[ticker].dropna().index, macd_line[ticker].dropna().values,
            label='MACD', linewidth=1.2, color='blue')
    ax2.plot(signal_line[ticker].dropna().index, signal_line[ticker].dropna().values,
            label='Signal', linewidth=1.2, color='orange')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_ylabel("MACD")
    ax2.legend(loc='upper left')


if __name__ == "__main__":
    constituents = get_constituents()
    df = retrieve_daily_close_prices_for_constituents(constituents)
    df = adjust_prices_for_splits_inferred(df, min_jump=0.12, max_den=20, tol=0.015)

    prices = df.copy()

    momentum_df = calculate_momentum_factor(df, lookback_period=90, exclude_recent_days=7)

    MODE = 'continuation'  # toggle: 'reversal' or 'continuation'

    reversals = screen_macd_reversals(
        prices,
        momentum_df,
        momentum_universe=75,
        crossover_window=7,
        mode=MODE,
    )

    group_a_df = reversals['group_a']
    group_b_df = reversals['group_b']

    if MODE == 'reversal':
        subject_label = "MACD Momentum Reversal"
        group_a_heading = "MACD Bearish Reversals — Top Momentum Stocks Turning Down"
        group_b_heading = "MACD Bullish Reversals — Bottom Momentum Stocks Turning Up"
        group_a_note = (
            "Top-75 momentum stocks where MACD is turning against the trend. "
            "Potential exits from longs or early short opportunities."
        )
        group_b_note = (
            "Bottom-75 momentum stocks where MACD is recovering against the downtrend. "
            "Potential covers of shorts or early long opportunities."
        )
    else:
        subject_label = "MACD Momentum Continuation"
        group_a_heading = "MACD Bullish Continuation — Top Momentum Stocks Resuming Uptrend"
        group_b_heading = "MACD Bearish Continuation — Bottom Momentum Stocks Resuming Downtrend"
        group_a_note = (
            "Top-75 momentum stocks that pulled back but whose MACD is now turning back up. "
            "Potential re-entry or add-to-long opportunities."
        )
        group_b_note = (
            "Bottom-75 momentum stocks that bounced but whose MACD is now turning back down. "
            "Potential re-entry or add-to-short opportunities."
        )

    print(f"\n--- {group_a_heading} ---")
    print(group_a_df.to_string() if not group_a_df.empty else "None found")

    print(f"\n--- {group_b_heading} ---")
    print(group_b_df.to_string() if not group_b_df.empty else "None found")

    mkt_caps = pd.read_excel(market_caps_path, sheet_name='Market Caps')
    mkt_caps = mkt_caps[["Ticker", 'Market Cap']].rename(columns={'Ticker': 'Symbol'}).set_index('Symbol')

    def format_market_cap(x):
        if pd.isna(x):
            return "—"
        x = float(x)
        for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
            if abs(x) >= div:
                return f"{x/div:,.2f}{unit}"
        return f"{x:,.0f}"

    def attach_market_cap(result_df):
        if result_df.empty:
            return result_df
        joined = result_df.join(mkt_caps, how='left')
        joined['Market Cap'] = joined['Market Cap'].map(format_market_cap)
        return joined

    def df_to_html(df, empty_msg="No signals today."):
        if df.empty:
            return f"<p>{empty_msg}</p>"
        return df.to_html(index=True, border=1, justify="center")

    group_a_df = attach_market_cap(group_a_df)
    group_b_df = attach_market_cap(group_b_df)

    from matplotlib.backends.backend_pdf import PdfPages

    macd_line, signal_line, histogram = calculate_macd(prices)

    pdf_path = os.path.join(plots_dir, f"macd_charts_{MODE}_{datetime.today().strftime('%Y-%m-%d')}.pdf")

    group_a_tickers = list(group_a_df.index)
    group_b_tickers = list(group_b_df.index)

    with PdfPages(pdf_path) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        ax.text(0.5, 0.6, f"{subject_label} Report", fontsize=22, ha='center', va='center', fontweight='bold')
        ax.text(0.5, 0.4, datetime.today().strftime('%Y-%m-%d'), fontsize=14, ha='center', va='center', color='gray')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Group A charts
        if group_a_tickers:
            fig, ax = plt.subplots(figsize=(12, 1))
            ax.axis('off')
            ax.text(0.5, 0.5, group_a_heading, fontsize=16, ha='center', va='center', fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            for ticker in group_a_tickers:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                                gridspec_kw={'height_ratios': [2, 1]})
                plot_macd(prices, ticker, macd_line, signal_line, histogram, ax1, ax2)
                signal_label = group_a_df.loc[ticker, 'Signal'] if ticker in group_a_df.index else ''
                momentum_label = group_a_df.loc[ticker, '3M-1W Momentum'] if ticker in group_a_df.index else ''
                fig.suptitle(f"{ticker}  |  {signal_label}  |  Momentum: {momentum_label}", fontsize=11, color='gray')
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        # Group B charts
        if group_b_tickers:
            fig, ax = plt.subplots(figsize=(12, 1))
            ax.axis('off')
            ax.text(0.5, 0.5, group_b_heading, fontsize=16, ha='center', va='center', fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            for ticker in group_b_tickers:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                                gridspec_kw={'height_ratios': [2, 1]})
                plot_macd(prices, ticker, macd_line, signal_line, histogram, ax1, ax2)
                signal_label = group_b_df.loc[ticker, 'Signal'] if ticker in group_b_df.index else ''
                momentum_label = group_b_df.loc[ticker, '3M-1W Momentum'] if ticker in group_b_df.index else ''
                fig.suptitle(f"{ticker}  |  {signal_label}  |  Momentum: {momentum_label}", fontsize=11, color='gray')
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"PDF saved: {pdf_path}")

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SENDER_EMAIL = "casselrobson93@gmail.com"
    SENDER_PASSWORD = "lajhhtqevwvomcts"
    RECIPIENT_EMAILS = ["casselrobson19@gmail.com", "misi2700@mylaurier.ca"]
    SUBJECT = f"Daily {subject_label} Report - {datetime.today().strftime('%Y-%m-%d')}"

    group_a_html = df_to_html(group_a_df, "No signals found.")
    group_b_html = df_to_html(group_b_df, "No signals found.")

    email_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            p.note {{ font-size: 0.85em; color: #555; }}
        </style>
    </head>
    <body>
        <h2>{group_a_heading}</h2>
        <p class="note">{group_a_note}</p>
        {group_a_html}
        <br/>
        <h2>{group_b_heading}</h2>
        <p class="note">{group_b_note}</p>
        {group_b_html}
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECIPIENT_EMAILS)
    msg["Subject"] = SUBJECT
    msg.attach(MIMEText(email_body, "html"))

    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(pdf_path)}")
        msg.attach(part)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
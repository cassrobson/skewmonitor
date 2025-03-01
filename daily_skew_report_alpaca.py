import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import warnings
warnings.filterwarnings('ignore')
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")

from alpaca.data.requests import OptionChainRequest
from alpaca.data.historical import OptionHistoricalDataClient
option_historical_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

def get_option_chain_snap(symbol):
    chain_snap = option_historical_data_client.get_option_chain(
    OptionChainRequest(
        underlying_symbol=symbol
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

def find_skew(df, expiry_dt):
    df = df[df['expiry_date']==expiry_dt]
    
    # Convert delta column to numeric, handling potential NaNs
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")

    # Separate calls and puts
    calls = df[df["option_type"] == "C"]
    puts = df[df["option_type"] == "P"]
    
    calls = calls.reset_index(drop=True)
    puts = puts.reset_index(drop=True)
    # Find the call with delta closest to 30
    closest_call = calls.iloc[(calls["delta"] - 0.30).abs().idxmin()]

    # Find the put with delta closest to -30
    closest_put = puts.iloc[(puts["delta"] + 0.30).abs().idxmin()]
    call_iv = closest_call['implied_volatility']
    put_iv = closest_put['implied_volatility']
    call_strike = closest_call['strike_price']
    put_strike = closest_put['strike_price']
    # Combine results into a single DataFrame
    skew = put_iv - call_iv

    return put_iv, call_iv, put_strike, call_strike, skew

def plot():
    plot_skew = pd.read_csv(r'C:\Users\Cassel Robson\skewmonitor\venv\skewmonitor\spx_skew.csv')
    plot_skew.columns = ['Date', '30Δ Call IV', "Call Strike", '30Δ Put IV', 'Put Strike', 'Skew']
    df = plot_skew
    df["Date"] = pd.to_datetime(df["Date"])

    # Plot

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Call IV and Put IV on primary y-axis
    ax1.plot(df["Date"], df["30Δ Call IV"], label="30Δ Call IV", marker='o', color='blue')
    ax1.plot(df["Date"], df["30Δ Put IV"], label="30Δ Put IV", marker='s', color='green')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("IV Values", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Create secondary y-axis for Skew
    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["Skew"], label="Skew", marker='^', color='red', linestyle='dashed')
    ax2.set_ylabel("Skew", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc="upper right")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.title("30Δ Call IV, 30Δ Put IV, and Skew Over Time")
    # Save plot
    plt.savefig(r"C:\Users\Cassel Robson\skewmonitor\venv\skewmonitor\plots\iv_skew_plot.png")
    plt.close()

    plot_skew = plot_skew.sort_values("Date", ascending=False)
    plot_skew['Change'] = plot_skew["Skew"]-plot_skew['Skew'].shift(-1)
    plot_skew["Signal"] = plot_skew["Change"].apply(lambda x: "Bearish" if x > 0 else "Bullish")
    today_signal = plot_skew.loc[len(plot_skew)-1, "Signal"]

    return plot_skew, today_signal
def track_daily_spx_skew():
    snap = get_option_chain_snap("SPY")

    expiry = get_next_third_friday()
    expiry_dt = pd.to_datetime(expiry, format="%Y-%m-%d").date()
    
    putiv, calliv, put_strike, call_strike, skew = find_skew(snap, expiry_dt)
    
    today = datetime.today().date()

    skew_df = pd.DataFrame({'Date': [today], 'C IV':[calliv], 'C Strike':[call_strike],'P IV':[putiv],'P Strike':[put_strike], 'SPY Skew': [skew]})

    try:
        skew_df.to_csv(r'C:\Users\Cassel Robson\skewmonitor\venv\skewmonitor\spx_skew.csv', mode='a', header=False, index=False)
    except FileNotFoundError:
        skew_df.to_csv(r'C:\Users\Cassel Robson\skewmonitor\venv\skewmonitor\spx_skew.csv', mode='w', header=True, index=False)

    df, signal = plot()
    return df, signal

if __name__=="__main__":
    df, signal = track_daily_spx_skew()
    plot_path = r"C:\Users\Cassel Robson\skewmonitor\venv\skewmonitor\plots\iv_skew_plot.png"  # Ensure this path is correct

    # Email configuration
    SMTP_SERVER = "smtp.gmail.com"  # Change based on your email provider
    SMTP_PORT = 587
    SENDER_EMAIL = "casselrobson93@gmail.com"
    SENDER_PASSWORD = "lajhhtqevwvomcts"  # Use an app password if using Gmail
    RECIPIENT_EMAIL = "casselrobson19@gmail.com"
    SUBJECT = f"Daily Skew Sentiment - {signal} - {datetime.today().strftime('%Y-%m-%d')}"

    # Convert DataFrame to HTML table
    df_html = df.to_html(index=False, border=1, justify="center")

    # Create email message
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
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
        <h2>Daily Skew Report (5D)</h2>
        {df_html}
        <p>Plot of Raw and Rolling Skew.</p>
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


    
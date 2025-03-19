import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
from email.mime.base import MIMEBase
from email import encoders
import warnings
import re
warnings.filterwarnings('ignore')
load_dotenv()

ALPACA_API_KEY="PKSWP8A9QH87I5DX69Y4"
ALPACA_API_SECRET="we0Fo1a7UmijE45Z3rWYkZjXqnsfFg3B8pwrnbdC"
ALPACA_BASE_URL="https://paper-api.alpaca.markets/"

from alpaca.data.requests import OptionChainRequest
from alpaca.data.historical import OptionHistoricalDataClient
option_historical_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

max_requests_per_minute = 150
delay_per_request = 60/max_requests_per_minute

script_dir = os.path.dirname(os.path.abspath(__file__))
skew_csv_path = os.path.join(script_dir, "constituent_skew.csv")

    
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

def find_skew(df, expiry_dt):
    expiry_dt = datetime(2025, 4, 17).date()
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

    # Combine results into a single DataFrame
    skew = put_iv - call_iv

    return skew

def get_constituents():
    sp500 = 'https://yfiua.github.io/index-constituents/constituents-sp500.csv'

    sp500 = pd.read_csv(sp500)
    sp500 = sp500.sort_values("Symbol", ascending=True)
    sp500_constituents = sp500['Symbol'].to_list()
    sp500_constituents = [x.replace("-",".") for x in sp500_constituents]
    sp500_constituents = [x for x in sp500_constituents if x != "BF.B"]

    return sp500_constituents

def largest_skew_changes(sp, error_symbols):
    skew = pd.read_csv(skew_csv_path)
    skew.columns = ['Date']+[x for x in sp if x not in error_symbols]
    skew = skew.set_index('Date')
    skew = skew - skew.shift(1)
    skew = skew.tail(1).transpose()
    print(skew)
    exit()
    skew.columns = ['Skew Change']
    skew = skew.sort_values('Skew Change')

    bearish = skew.tail(3).sort_values('Skew Change', ascending=False)
    bullish = skew.head(3)

    return bearish, bullish

def track_daily_constituent_skew():
    sp = get_constituents()
    data = {}
    error_symbols = []
    for i, symbol in enumerate(sp, start=1):
        try:
            print(symbol)
            snap = get_option_chain_snap(symbol)

            expiry = get_next_third_friday()
            expiry_dt = pd.to_datetime(expiry, format="%Y-%m-%d").date()
            
            skew = find_skew(snap, expiry_dt)
            data[symbol] = skew

        except Exception as e:
            print("Error for {}: {}".format(symbol, e))
            error_symbols.append(symbol)
            continue

        if i % max_requests_per_minute == 0:
            print("Reached 150 API calls, sleeping for 60 seconds")
            time.sleep(60)

    
    today = datetime.today().date()
    skew_df = pd.DataFrame(data, index=[today])
    
    try:
        skew_df.to_csv(skew_csv_path, mode='a', header=False, index=True)
    except FileNotFoundError:
        skew_df.to_csv(skew_csv_path, mode='w', header=True, index=True)
    
    bearish, bullish = largest_skew_changes(sp, error_symbols)
    
    return bearish, bullish

if __name__=="__main__":
    bearish, bullish = track_daily_constituent_skew()
    # Email configuration
    # Email configuration
    SMTP_SERVER = "smtp.gmail.com"  # Gmail SMTP server
    SMTP_PORT = 587
    SENDER_EMAIL = "casselrobson93@gmail.com"
    SENDER_PASSWORD = "lajhhtqevwvomcts"  
    RECIPIENT_EMAILS = ["casselrobson19@gmail.com", "misi2700@mylaurier.ca"]
    SUBJECT = f"Daily Constituent Skew - {datetime.today().strftime('%Y-%m-%d')}"

    # Convert DataFrame to HTML table
    bearish_html = bearish.to_html(index=True, border=1, justify="center")
    bullish_html = bullish.to_html(index=True, border=1, justify="center")

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
        <h2>Bearish names, Put IV Up, Short Expensive Puts / Long Cheap Calls (5D)</h2>
        {bearish_html}
        <h2>Bullish names, Call IV Up, Long Cheap Puts / Short Expensive Calls (5D)</h2>
        {bullish_html}
    </body>
    </html>
    """

    msg.attach(MIMEText(email_body, "html"))

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


    
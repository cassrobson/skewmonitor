import yfinance as yf
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
import requests
import math
import getpass

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import scipy.stats as stats
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

# Function to fetch SPX options data
def fetch_spx_options_data():
    spx = yf.Ticker('^SPX')
    
    # Get options expiration dates for SPX
    expiration_dates = spx.options
    # Fetch the options data for the nearest expiration date (or specify one)
    exp = expiration_dates[5]
    options_data = spx.option_chain(exp)  # Choose the nearest expiration

    # Combine the calls and puts into one DataFrame
    calls = options_data.calls

    puts = options_data.puts
    
    return calls, puts, exp

def get_spx_price():
    spx = yf.Ticker('^SPX')
    spx_price = spx.history(period='1d')['Close'][0]  # Get the latest closing price
    return spx_price

def calculate_call_delta(S, K, T, r, sigma):
    # Calculate d1
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    # Call delta is the cumulative normal distribution function of d1
    delta = stats.norm.cdf(d1)
    return delta

# Function to calculate SPX skew
def calculate_spx_skew(calls, puts, spot, T):
    spotr = round(spot, 0)
    calls = calls[(calls['strike'] <= spotr+100)&(calls['strike']>= spotr-100)]
    puts = puts[(puts['strike'] <= spotr+100)&(puts['strike']>= spotr-100)]
    calls['delta'] = calls.apply(lambda row: calculate_call_delta(spot, row['strike'], T, 0, row['impliedVolatility']), axis=1)
    puts['delta'] = puts.apply(lambda row: calculate_call_delta(spot, row['strike'], T, 0, row['impliedVolatility']), axis=1)

    calls = calls[(calls['delta']>=0.25)&(calls['delta']<=0.3)].head(1).reset_index()
    puts = puts[(puts['delta']>=0.25)&(puts['delta']<=0.3)].head(1).reset_index()

    call_30d_iv = calls.loc[0, "impliedVolatility"]
    put_30d_iv = puts.loc[0, "impliedVolatility"]
    skew = put_30d_iv-call_30d_iv

    return call_30d_iv, put_30d_iv, skew

def plot():
    plot_skew = pd.read_csv("spx_skew.csv")
    plot_skew.columns = ['Date', '30Δ Call IV', '30Δ Put IV', 'Skew']
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
    plt.savefig("plots/iv_skew_plot.png", bbox_inches="tight")
    plt.close()
    return plot_skew


# Function to track daily SPX skew
def track_daily_spx_skew():
    # Get current date
    today = datetime.today()


    spot = get_spx_price()

    
    # Fetch SPX options data
    calls, puts, exp = fetch_spx_options_data()

    time_to_exp = datetime.strptime(exp, "%Y-%m-%d") + timedelta(hours=16)
    time_to_exp = time_to_exp-today
    time_to_exp_sec = time_to_exp.total_seconds()
    time_to_exp = time_to_exp_sec / (252*24*60*60)

    # Calculate SPX skew
    call_30d_iv, put_30d_iv, skew = calculate_spx_skew(calls, puts, spot, time_to_exp)
    
    # Store the result in a DataFrame
    skew_df = pd.DataFrame({'Date': [today], '30Δ Call IV':[call_30d_iv], '30Δ Put IV':[put_30d_iv], 'SPX_Skew': [skew]})
  
    # Save to CSV (append mode)
    try:
        skew_df.to_csv('spx_skew.csv', mode='a', header=False, index=False)
    except FileNotFoundError:
        skew_df.to_csv('spx_skew.csv', mode='w', header=True, index=False)

    df = plot()
    

    print(f"SPX Skew on {today}: {skew}")
    return df

def slack(df):
    # Slack API Configuration
    SLACK_BOT_TOKEN = "xoxb-8442501448374-8449112677459-9jKEx9UUK0mdyfK8xLWDSPVy"
    SLACK_CHANNEL_ID = "C08CSHKM947"  # Replace with your channel ID

    # Convert DataFrame to Markdown for Slack
    df_markdown = df.to_markdown(index=False)

    # Send a message with the DataFrame
    slack_message = f"*📊 Daily Report - {pd.Timestamp.today().strftime('%Y-%m-%d')}*\n```{df_markdown}```"

    response = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        json={"channel": SLACK_CHANNEL_ID, "text": slack_message},
    )

    if response.json().get("ok"):
        print("Message sent successfully to Slack!")
    else:
        print("Error sending message:", response.json())

# Run the script daily
if __name__ == "__main__":
    df = track_daily_spx_skew()
    plot_path = "plots/iv_skew_plot.png"  # Ensure this path is correct

    # Email configuration
    SMTP_SERVER = "smtp.gmail.com"  # Change based on your email provider
    SMTP_PORT = 587
    SENDER_EMAIL = "casselrobson93@gmail.com"
    SENDER_PASSWORD = "lajhhtqevwvomcts"  # Use an app password if using Gmail
    RECIPIENT_EMAIL = "casselrobson19@gmail.com"
    SUBJECT = f"Daily Report - {datetime.today().strftime('%Y-%m-%d')}"

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
        <h2>Daily Data Report</h2>
        {df_html}
        <p>Attached is the latest plot.</p>
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
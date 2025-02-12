import yfinance as yf
import pandas as pd
import warnings
import math
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
    print(calls)
    print(puts)
    exit()
    
    # Filter out-of-the-money (OTM) puts and calls
    atm_strike = (calls['strike'] + puts['strike']) / 2

    oitm_calls = calls[calls['strike'] > atm_strike]
    oitm_puts = puts[puts['strike'] < atm_strike]
    
    # Calculate implied volatility difference (skew)
    iv_calls_avg = oitm_calls['impliedVolatility'].mean()
    iv_puts_avg = oitm_puts['impliedVolatility'].mean()
    
    skew = iv_puts_avg - iv_calls_avg
    return skew

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
    skew = calculate_spx_skew(calls, puts, spot, time_to_exp)
    
    # Store the result in a DataFrame
    skew_df = pd.DataFrame({'Date': [today], 'SPX_Skew': [skew]})
    
    # Save to CSV (append mode)
    try:
        skew_df.to_csv('spx_skew.csv', mode='a', header=False, index=False)
    except FileNotFoundError:
        skew_df.to_csv('spx_skew.csv', mode='w', header=True, index=False)

    print(f"SPX Skew on {today}: {skew}")

# Run the script daily
if __name__ == "__main__":
    track_daily_spx_skew()

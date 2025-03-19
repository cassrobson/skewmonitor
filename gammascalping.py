import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
from ib_insync import *
warnings.filterwarnings('ignore')

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=9)

def get_price(ticker):
    price  = ticker.history(period='1d')['Close'][0]
    return price

def fetch_options_data(ticker, expiration):
    options_data = ticker.option_chain(expiration)
    calls = options_data.calls
    return calls

def get_position(symbol):
    # Request positions
    positions = ib.positions()
    
    # Find the position size for the specified ticker, ensuring it's an equity position
    position_size = next((pos.position for pos in positions if pos.contract.symbol == symbol and pos.contract.secType == "STK"), 0)
    
    return position_size

def place_market_order(symbol, quantity):
    if quantity == 0:
        print(f"No need to trade. Position already hedged for {symbol}.")
        return

    action = "BUY" if quantity > 0 else "SELL"
    contract = Stock(symbol, "SMART", "USD")  # Define stock contract
    order = MarketOrder(action, abs(quantity))  # Create market order

    trade = ib.placeOrder(contract, order)  # Send the order

    # Debugging: Print order details
    print(f"📌 Placing {action} market order for {abs(quantity)} shares of {symbol}...")

    ib.sleep(2)  # Wait for IBKR to process the order
    print(f"🚀 Order status: {trade.orderStatus.status}")

    if trade.orderStatus.status not in ["Submitted", "Filled"]:
        print(f"⚠️ Order issue! Status: {trade.orderStatus.status}")
        print(f"Trade log: {trade.log}")
    else:
        print(f"✅ Order successfully submitted!")

def parse_option_input(option_string):
    """Parses an option string like '1 SMCI 48 C20250620' and converts expiration to 'YYYY-MM-DD'."""
    try:
        parts = option_string.split()

        contracts = int(parts[0])  # Number of contracts
        symbol = parts[1]  # Ticker symbol
        strike = float(parts[2])  # Strike price

        option_type = parts[3][0]  # 'C' for Call or 'P' for Put
        expiration_raw = parts[3][1:]  # Extract YYYYMMDD

        # Convert YYYYMMDD → YYYY-MM-DD
        expiration = datetime.strptime(expiration_raw, "%Y%m%d").strftime("%Y-%m-%d")

        return contracts, symbol, strike, option_type, expiration

    except (IndexError, ValueError) as e:
        print("⚠️ Invalid input format! Use: 1 SMCI 48 C20250620")
        return None
    
if __name__ == "__main__":
    contract = input("Contract (Ex. 1 SMCI 48 C20250620): ")

    contracts, symbol, strike, option_type, expiration = parse_option_input(contract)

    today_delta = float(input("Current Delta: "))
    
    today = datetime.today()
    
    ticker = yf.Ticker(symbol)
    spot = get_price(ticker)
    
    shares_owned = get_position(symbol)
    
    
    hedge = today_delta * contracts * 100 * -1
    
    
    shares_to_order = hedge - shares_owned

    place_market_order(symbol, round(shares_to_order,0))
    





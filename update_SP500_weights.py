import pandas as pd
import yfinance as yf

def get_constituents():
    sp500 = 'https://yfiua.github.io/index-constituents/constituents-sp500.csv'

    sp500 = pd.read_csv(sp500)
    sp500 = sp500.sort_values("Symbol", ascending=True)
    sp500_constituents = sp500['Symbol'].to_list()
    sp500_constituents = [x.replace(".","-") for x in sp500_constituents]

    return sp500_constituents
def update_weights(sp):
    data = []

    for symbol in sp:
        try:
            ticker = yf.Ticker(symbol)
            market_cap = ticker.info.get('marketCap')  # Use .get() to avoid KeyError if missing
            if market_cap is None:
                print(f"Market cap not found for {symbol}. Skipping.")
                continue
            data.append({"symbol": symbol, "market_cap": market_cap})
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    df = pd.DataFrame(data)
    df.to_pickle(r"C:\Users\Cassel Robson\skewmonitor\venv\skewmonitor\market_caps\sp500_market_caps.pkl")  # Save as a pickle file
    print("Data saved to sp500_market_caps.pkl")


if __name__ == "__main__":
    sp = get_constituents()
    df = pd.DataFrame({'Symbol': sp})
    df.to_excel('market_caps.xlsx')
    exit()

    update_weights(sp[:3])
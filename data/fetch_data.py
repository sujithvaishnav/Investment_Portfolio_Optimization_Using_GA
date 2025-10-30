import yfinance as yf
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_TICKERS = [
    "SAN.MC", "BBVA.MC", "ITX.MC", "TEF.MC", "IBE.MC", "FER.MC",
    "ACS.MC", "REP.MC", "GRF.MC", "BKT.MC", "CLNX.MC", "CABK.MC",
    "AENA.MC", "AMS.MC", "SGRE.MC", "MEL.MC", "VIS.MC", "REE.MC",
    "COL.MC", "MAP.MC", "ENG.MC", "ELE.MC", "ALM.MC", "MRL.MC",
    "PHM.MC", "CIE.MC", "BME.MC", "SAB.MC", "MTS.MC", "A3M.MC",
    "CLH.MC", "ACX.MC"
]

def fetch_ibex35_data(tickers=DEFAULT_TICKERS, start="2017-07-01", end="2025-04-26"):
    """
    Fetch Close prices and compute daily returns.
    Saves CSVs into data/ folder and returns (prices_df, returns_df).
    """
    print("📡 Downloading tickers:", tickers)
    data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker')

    # Handle multi-ticker data
    prices = pd.DataFrame()
    for t in tickers:
        try:
            prices[t] = data[t]['Close']
        except Exception:
            print(f"⚠️ Missing data for {t}")
            continue

    # Drop empty columns
    dropped = [c for c in prices.columns if prices[c].isna().all()]
    if dropped:
        print("🚫 Dropped tickers with no data:", dropped)
        prices = prices.drop(columns=dropped)

    # Save prices
    prices.to_csv(os.path.join(DATA_DIR, "ibex35_close.csv"))
    print(f"✅ Saved ibex35_close.csv with shape {prices.shape}")

    # Compute returns
    returns = prices.pct_change().dropna(how="all")
    returns.to_csv(os.path.join(DATA_DIR, "ibex35_returns.csv"))
    print(f"✅ Saved ibex35_returns.csv with shape {returns.shape}")

    return prices, returns


def load_ibex35_data():
    """Load from CSV if present, otherwise fetch."""
    close_path = os.path.join(DATA_DIR, "ibex35_close.csv")
    returns_path = os.path.join(DATA_DIR, "ibex35_returns.csv")

    if os.path.exists(close_path) and os.path.exists(returns_path):
        prices = pd.read_csv(close_path, index_col=0, parse_dates=True)
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded existing data: prices {prices.shape}, returns {returns.shape}")
        return prices, returns
    else:
        return fetch_ibex35_data()


if __name__ == "__main__":
    p, r = fetch_ibex35_data()
    print("📈 Prices shape:", p.shape)
    print("📉 Returns shape:", r.shape)

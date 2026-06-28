import pandas as pd
import yfinance as yf


def download_adjusted_close_prices(tickers, start_date, end_date=None):
    """
    Download adjusted close prices from Yahoo Finance.

    Parameters:
        tickers: list of ticker symbols
        start_date: start date in YYYY-MM-DD format
        end_date: optional end date in YYYY-MM-DD format

    Returns:
        pandas DataFrame of adjusted close prices
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No price data downloaded. Check tickers or date range.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all")
    return prices
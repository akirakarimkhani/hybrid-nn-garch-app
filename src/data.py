"""
Data loading and preprocessing for the Streamlit app.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


def load_stock_data(ticker: str, start: str, end: str):
    """
    Download OHLCV data and compute log returns × 100.
    Returns (prices: Series, returns: Series).
    """
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. "
                         "Check the ticker symbol and date range.")
    close = data["Close"].squeeze()
    log_ret = np.log(close / close.shift(1)).dropna() * 100
    return close, log_ret


def train_test_split_returns(returns: pd.Series, ratio: float = 0.8):
    split = int(len(returns) * ratio)
    return returns.iloc[:split], returns.iloc[split:]


def descriptive_stats(returns: pd.Series) -> dict:
    """Return key descriptive statistics for the returns series."""
    r = returns.dropna()
    adf_res = adfuller(r, maxlag=10, autolag=None)
    lb = acorr_ljungbox(r, lags=[10], return_df=True)
    lb_sq = acorr_ljungbox(r ** 2, lags=[10], return_df=True)

    return {
        "Observations": len(r),
        "Mean": float(np.round(r.mean(), 4)),
        "Std Dev": float(np.round(r.std(), 4)),
        "Min": float(np.round(r.min(), 4)),
        "Max": float(np.round(r.max(), 4)),
        "Skewness": float(np.round(stats.skew(r), 4)),
        "Kurtosis (excess)": float(np.round(stats.kurtosis(r), 4)),
        "ADF Stat": float(np.round(adf_res[0], 4)),
        "ADF p-value": float(np.round(adf_res[1], 4)),
        "Ljung-Box(10) stat": float(np.round(lb["lb_stat"].iloc[0], 4)),
        "Ljung-Box(10) p-val": float(np.round(lb["lb_pvalue"].iloc[0], 4)),
        "LB²(10) stat": float(np.round(lb_sq["lb_stat"].iloc[0], 4)),
        "LB²(10) p-val": float(np.round(lb_sq["lb_pvalue"].iloc[0], 4)),
    }

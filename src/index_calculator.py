import numpy as np
import pandas as pd


def calculate_daily_returns(prices):
    """
    Calculate daily percentage returns from price levels.
    """
    return prices.pct_change().dropna(how="all")


def calculate_equal_weight_basket_returns(asset_returns):
    """
    Calculate equal-weighted basket returns.

    In this first version, the basket is rebalanced daily back to equal weights.
    Later, we will improve this to monthly rebalancing.
    """
    number_of_assets = len(asset_returns.columns)

    weights = pd.Series(
        1 / number_of_assets,
        index=asset_returns.columns
    )

    basket_returns = asset_returns.mul(weights, axis=1).sum(axis=1)
    return basket_returns


def calculate_realized_volatility(returns, lookback_days, annualization_factor):
    """
    Calculate annualized realized volatility using rolling daily returns.
    """
    realized_volatility = returns.rolling(lookback_days).std() * np.sqrt(annualization_factor)
    return realized_volatility


def calculate_volatility_target_exposure(
    realized_volatility,
    target_volatility,
    min_exposure,
    max_exposure
):
    """
    Calculate volatility-targeted exposure.

    Exposure = Target Volatility / Realized Volatility

    Exposure is then capped and floored.
    """
    raw_exposure = target_volatility / realized_volatility
    final_exposure = raw_exposure.clip(lower=min_exposure, upper=max_exposure)
    return final_exposure


def calculate_index_level(returns, starting_index_level):
    """
    Convert daily returns into an index level.
    """
    index_level = starting_index_level * (1 + returns).cumprod()
    return index_level
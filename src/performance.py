import numpy as np
import pandas as pd


def calculate_performance_metrics(returns, annualization_factor=252):
    """
    Calculate standard performance metrics for a daily return series.

    Metrics:
    - Annualized return
    - Annualized volatility
    - Sharpe ratio
    - Maximum drawdown
    - Cumulative return
    """
    returns = returns.dropna()

    if returns.empty:
        raise ValueError("Return series is empty. Cannot calculate performance metrics.")

    cumulative_return = (1 + returns).prod() - 1

    annualized_return = (1 + cumulative_return) ** (annualization_factor / len(returns)) - 1

    annualized_volatility = returns.std() * np.sqrt(annualization_factor)

    if annualized_volatility != 0:
        sharpe_ratio = annualized_return / annualized_volatility
    else:
        sharpe_ratio = np.nan

    index_level = (1 + returns).cumprod()
    running_max = index_level.cummax()
    drawdown = index_level / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }


def build_performance_summary(results, annualization_factor=252):
    """
    Build performance summary for basket, volatility-targeted index, and benchmark.
    """
    performance_summary = pd.DataFrame({
        "Equal Weight Basket": calculate_performance_metrics(
            results["Basket Return"],
            annualization_factor
        ),
        "Vol Target Index": calculate_performance_metrics(
            results["Vol Target Return"],
            annualization_factor
        ),
        "Benchmark": calculate_performance_metrics(
            results["Benchmark Return"],
            annualization_factor
        ),
    })

    return performance_summary
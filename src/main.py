import pandas as pd

from config import (
    TICKERS,
    BENCHMARK,
    START_DATE,
    END_DATE,
    TARGET_VOLATILITY,
    VOLATILITY_LOOKBACK_DAYS,
    MAX_EXPOSURE,
    MIN_EXPOSURE,
    ANNUALIZATION_FACTOR,
    STARTING_INDEX_LEVEL,
)

from data_loader import download_adjusted_close_prices

from index_calculator import (
    calculate_daily_returns,
    calculate_equal_weight_basket_returns,
    calculate_realized_volatility,
    calculate_volatility_target_exposure,
    calculate_index_level,
)


def main():
    all_tickers = list(set(TICKERS + [BENCHMARK]))

    prices = download_adjusted_close_prices(
        tickers=all_tickers,
        start_date=START_DATE,
        end_date=END_DATE
    )

    asset_prices = prices[TICKERS]
    benchmark_prices = prices[BENCHMARK]

    asset_returns = calculate_daily_returns(asset_prices)
    benchmark_returns = calculate_daily_returns(benchmark_prices.to_frame())[BENCHMARK]

    basket_returns = calculate_equal_weight_basket_returns(asset_returns)

    realized_volatility = calculate_realized_volatility(
        returns=basket_returns,
        lookback_days=VOLATILITY_LOOKBACK_DAYS,
        annualization_factor=ANNUALIZATION_FACTOR
    )

    exposure = calculate_volatility_target_exposure(
        realized_volatility=realized_volatility,
        target_volatility=TARGET_VOLATILITY,
        min_exposure=MIN_EXPOSURE,
        max_exposure=MAX_EXPOSURE
    )

    # Use previous day's exposure for today's return to avoid look-ahead bias.
    shifted_exposure = exposure.shift(1)

    volatility_targeted_returns = shifted_exposure * basket_returns

    results = pd.DataFrame({
        "Basket Return": basket_returns,
        "Realized Volatility": realized_volatility,
        "Exposure": exposure,
        "Shifted Exposure": shifted_exposure,
        "Vol Target Return": volatility_targeted_returns,
        "Benchmark Return": benchmark_returns,
    }).dropna()

    results["Basket Index"] = calculate_index_level(
        returns=results["Basket Return"],
        starting_index_level=STARTING_INDEX_LEVEL
    )

    results["Vol Target Index"] = calculate_index_level(
        returns=results["Vol Target Return"],
        starting_index_level=STARTING_INDEX_LEVEL
    )

    results["Benchmark Index"] = calculate_index_level(
        returns=results["Benchmark Return"],
        starting_index_level=STARTING_INDEX_LEVEL
    )

    print(results.tail())
    print("\nProject run completed successfully.")


if __name__ == "__main__":
    main()
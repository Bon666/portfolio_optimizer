import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def ensure_dirs():
    """
    Create output and data directories if they do not exist.
    """
    out_dir = Path("outputs")
    data_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    return out_dir, data_dir


def fetch_prices(tickers, years, data_dir):
    """
    Fetch adjusted close prices from Yahoo Finance.
    Data is cached locally for reproducibility.
    """
    tickers = [t.upper() for t in tickers]
    cache_path = data_dir / f"{'-'.join(tickers)}_{years}y.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")

    prices = yf.download(
        tickers,
        period=f"{years}y",
        auto_adjust=True,
        progress=False
    )["Close"]

    prices.index.name = "Date"
    prices.to_csv(cache_path)
    return prices


def portfolio_stats(weights, mu, cov, rf):
    """
    Compute annualized return, volatility, and Sharpe ratio.
    """
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(weights.T @ cov @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else -np.inf
    return ret, vol, sharpe


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Optimizer using Monte Carlo simulation"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "AMZN"],
        help="List of stock tickers"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of historical data"
    )
    parser.add_argument(
        "--rf",
        type=float,
        default=0.0,
        help="Annual risk-free rate (e.g., 0.02 for 2%)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8000,
        help="Number of random portfolios"
    )

    args = parser.parse_args()

    out_dir, data_dir = ensure_dirs()

    # Fetch and prepare data
    prices = fetch_prices(args.tickers, args.years, data_dir).dropna()
    if prices.empty:
        raise RuntimeError("No price data available. Check tickers or internet connection.")

    returns = prices.pct_change().dropna()

    # Annualized expected returns and covariance matrix
    mu = returns.mean() * 252
    cov = returns.cov() * 252

    results = []
    weights_list = []

    rng = np.random.default_rng(42)
    num_assets = len(mu)

    # Monte Carlo simulation
    for _ in range(args.n):
        weights = rng.random(num_assets)
        weights /= weights.sum()

        r, v, s = portfolio_stats(weights, mu.values, cov.values, args.rf)
        results.append((r, v, s))
        weights_list.append(weights)

    results = np.array(results)
    weights_list = np.array(weights_list)

    # Identify optimal portfolios
    max_sharpe_idx = int(np.argmax(results[:, 2]))
    min_vol_idx = int(np.argmin(results[:, 1]))

    max_sharpe_weights = weights_list[max_sharpe_idx]
    min_vol_weights = weights_list[min_vol_idx]

    # Save portfolio allocations
    pd.Series(
        max_sharpe_weights,
        index=mu.index,
        name="weight"
    ).sort_values(ascending=False).to_csv(
        out_dir / "allocation_max_sharpe.csv"
    )

    pd.Series(
        min_vol_weights,
        index=mu.index,
        name="weight"
    ).sort_values(ascending=False).to_csv(
        out_dir / "allocation_min_vol.csv"
    )

    # Plot Efficient Frontier
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        results[:, 1],
        results[:, 0],
        c=results[:, 2],
        cmap="viridis",
        s=8,
        alpha=0.9
    )
    plt.colorbar(scatter, label="Sharpe Ratio")

    plt.scatter(
        results[max_sharpe_idx, 1],
        results[max_sharpe_idx, 0],
        marker="*",
        s=300,
        color="red",
        label="Max Sharpe"
    )
    plt.scatter(
        results[min_vol_idx, 1],
        results[min_vol_idx, 0],
        marker="X",
        s=150,
        color="black",
        label="Min Volatility"
    )

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier (Monte Carlo)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_dir / "efficient_frontier.png", dpi=200)
    plt.show()

    # Console summary
    print("Results saved to:", out_dir.resolve())
    print("Max Sharpe Ratio:", round(float(results[max_sharpe_idx, 2]), 4))
    print("Min Volatility:", round(float(results[min_vol_idx, 1]), 4))


if __name__ == "__main__":
    main()

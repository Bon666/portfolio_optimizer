# Portfolio Optimizer

📈 A Python-based tool to optimize a portfolio of stocks by maximizing the Sharpe Ratio and minimizing risk using Monte Carlo simulation and numerical optimization.

## ✅ Features

- Downloads historical price data from Yahoo Finance
- Calculates annualized return and volatility
- Simulates thousands of random portfolios
- Visualizes risk-return space with Sharpe ratio coloring
- Identifies optimal portfolios:
  - Maximum Sharpe Ratio
  - Minimum Volatility

## 💡 Stocks Used

- AAPL, MSFT, GOOGL, AMZN, META (modifiable)

## ▶️ How to Run

```bash
pip install -r requirements.txt
python portfolio_optimizer.py
```

## 📂 Output

- `graphs/monte_carlo_portfolios.png`: Risk-return scatter plot of portfolios
- `stock_prices.csv`: Price data file

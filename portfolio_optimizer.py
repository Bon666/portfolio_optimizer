import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import os

# Define the stocks and download data
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
data = yf.download(stocks, start="2022-01-01", end="2024-12-31")["Adj Close"]
data.to_csv("stock_prices.csv")

# Calculate returns and statistics
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_portfolios = 5000
risk_free_rate = 0.0175

# Monte Carlo Simulation
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    weights_record.append(weights)
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    results[0,i] = portfolio_return
    results[1,i] = portfolio_stddev
    results[2,i] = (portfolio_return - risk_free_rate) / portfolio_stddev

# Extract optimal portfolios
max_sharpe_idx = np.argmax(results[2])
min_volatility_idx = np.argmin(results[1])
optimal_weights = weights_record[max_sharpe_idx]

# Plot results
plt.figure(figsize=(12, 8))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', alpha=0.7)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(results[1,max_sharpe_idx], results[0,max_sharpe_idx], c='red', marker='*', s=200, label='Max Sharpe Ratio')
plt.scatter(results[1,min_volatility_idx], results[0,min_volatility_idx], c='blue', marker='X', s=200, label='Min Volatility')
plt.title('Efficient Frontier with Monte Carlo Simulation')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Return')
plt.legend()
plt.tight_layout()
os.makedirs("graphs", exist_ok=True)
plt.savefig("graphs/monte_carlo_portfolios.png")
plt.show()

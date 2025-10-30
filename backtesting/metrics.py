import numpy as np
import pandas as pd

def portfolio_returns(weights, returns_df):
    return returns_df.dot(weights)

def average_daily_return(port_returns):
    return float(port_returns.mean())

def volatility(port_returns):
    return float(port_returns.std(ddof=0))

def max_drawdown(port_returns):
    cum = (1 + port_returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return float(drawdown.min())

def cvar(port_returns, alpha=0.05):
    var = np.percentile(port_returns, alpha*100)
    return float(port_returns[port_returns <= var].mean())

def evaluate_portfolio(weights, returns_df, benchmark=False):
    if benchmark:
        weights = np.ones(returns_df.shape[1]) / returns_df.shape[1]
    port_returns = portfolio_returns(weights, returns_df)
    return {
        "Average Daily Return": average_daily_return(port_returns),
        "Volatility": volatility(port_returns),
        "Max Drawdown": max_drawdown(port_returns),
        "5% CVaR": cvar(port_returns, alpha=0.05)
    }

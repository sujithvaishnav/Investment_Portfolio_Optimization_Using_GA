import numpy as np
import pandas as pd

def portfolio_metrics(weights, returns_df):
    """
    Compute portfolio metrics: mean return, volatility, skewness, kurtosis.
    Handles divide-by-zero safely.
    """
    port_daily = (returns_df * weights).sum(axis=1)

    mu = port_daily.mean()
    sigma = port_daily.std(ddof=0)
    skew_p = port_daily.skew()
    kurt_p = port_daily.kurtosis()

    # Prevent division by zero
    if sigma == 0 or np.isnan(sigma):
        sigma = 1e-8

    return mu, sigma, skew_p, kurt_p


import numpy as np

def fitness_function(weights, returns_df,
                     alpha=0.5, beta=0.3, gamma=0.1, delta=0.1,
                     max_weight=0.5, esg_tickers=None):
    # Ensure weights is flat array
    weights = np.array(weights, dtype=float).flatten()

    mu, sigma, skew_p, kurt_p = portfolio_metrics(weights, returns_df)

    # Base fitness (to minimize)
    fitness = -(alpha * mu - beta * sigma + gamma * skew_p - delta * kurt_p)

    # Constraint penalties
    if np.any(weights < 0) or np.any(weights > max_weight):
        fitness += 1000 * np.sum(np.clip(weights - max_weight, 0, None) + np.clip(-weights, 0, None))
    
    sum_diff = abs(np.sum(weights) - 1.0)
    if sum_diff > 1e-3:
        fitness += 500 * sum_diff

    # ESG bonus
    if esg_tickers is not None and len(esg_tickers) > 0:
        ticker_to_index = {ticker: idx for idx, ticker in enumerate(returns_df.columns)}
        esg_indices = [ticker_to_index[t] for t in esg_tickers if t in ticker_to_index]
        esg_weight = sum(weights[i] for i in esg_indices)
        fitness -= 0.02 * esg_weight

    # Return single tuple of Python float
    return (float(fitness),)




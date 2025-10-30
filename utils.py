import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_weights(weights):
    w = np.array(weights, dtype=float)
    total = w.sum()
    if total == 0:
        return np.ones_like(w) / len(w)
    return w / total

def save_portfolio_csv(weights, tickers, filename="optimized_portfolio.csv"):
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    df.to_csv(filename, index=False)
    print(f"Saved portfolio to {filename}")
    return filename

def plot_weights_matplotlib(weights, tickers, title="Portfolio Weights"):
    plt.figure(figsize=(12, 5))
    plt.bar(tickers, weights)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_fitness_matplotlib(history, title="Fitness Convergence"):
    plt.figure(figsize=(10, 4))
    plt.plot(history, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.grid(True)
    return plt.gcf()

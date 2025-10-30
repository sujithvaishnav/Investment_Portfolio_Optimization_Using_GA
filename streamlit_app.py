import streamlit as st
import pandas as pd
from data.fetch_data import load_ibex35_data
from ga.run_ga import run_ga
from backtesting.metrics import evaluate_portfolio
from utils import plot_weights_matplotlib, plot_fitness_matplotlib, save_portfolio_csv
import matplotlib.pyplot as plt

st.set_page_config(page_title="IBEX35 GA Portfolio", layout="wide")
st.title("IBEX35 Portfolio Optimization (GA)")

# Load data
prices, returns = load_ibex35_data()
tickers = list(returns.columns)

st.sidebar.header("Data & Timeframe")
st.sidebar.write(f"Data rows: {returns.shape[0]}, assets: {returns.shape[1]}")

# GA parameters
st.sidebar.header("GA Parameters")
num_generations = st.sidebar.slider("Number of generations", 5, 200, 50, 5)
pop_size = st.sidebar.slider("Population size", 50, 1000, 660, 10)
num_parents = st.sidebar.slider("Number of parents", 10, 500, 150, 10)
crossover_prob = st.sidebar.slider("Crossover probability", 0.0, 1.0, 0.5, 0.05)
mutation_prob = st.sidebar.slider("Mutation probability", 0.0, 1.0, 0.4, 0.05)
indpb = st.sidebar.slider("Per-gene mutation prob (indpb)", 0.0, 1.0, 0.05, 0.01)
max_weight = st.sidebar.slider("Max weight per asset", 0.0, 1.0, 0.5, 0.05)

st.sidebar.header("COF parameters")
alpha = st.sidebar.number_input("Alpha (return)", 0.0, 10.0, 1.0, 0.1)
beta = st.sidebar.number_input("Beta (volatility)", 0.0, 10.0, 1.0, 0.1)
gamma = st.sidebar.number_input("Gamma (skew)", 0.0, 10.0, 0.0, 0.1)
delta = st.sidebar.number_input("Delta (kurtosis)", 0.0, 10.0, 0.0, 0.1)

st.sidebar.header("ESG")
use_esg = st.sidebar.checkbox("Apply ESG bonus", value=True)
esg_tickers = ["SAN.MC", "BBVA.MC", "ITX.MC", "TEF.MC", "IBE.MC"] if use_esg else None

run_button = st.button("Run GA Optimization")

if run_button:
    with st.spinner("Running GA... (start with small settings for testing)"):
        best_weights, fitness_history = run_ga(
            returns_df=returns,
            num_generations=num_generations,
            pop_size=pop_size,
            num_parents=num_parents,
            max_weight=max_weight,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            crossover_prob=crossover_prob, mutation_prob=mutation_prob, indpb=indpb,
            esg_tickers=esg_tickers,
            seed=42
        )

    st.success("Optimization finished!")

    # Display weights table
    df_w = pd.DataFrame({"Ticker": tickers, "Weight": best_weights})
    st.subheader("Optimized Weights")
    st.dataframe(df_w.style.format({"Weight": "{:.4f}"}))

    # Plot weights
    fig1 = plot_weights_matplotlib(best_weights, tickers)
    st.pyplot(fig1)

    # Fitness history
    st.subheader("Fitness Convergence")
    fig2 = plot_fitness_matplotlib(fitness_history)
    st.pyplot(fig2)

    # Metrics
    st.subheader("Performance vs Equal-weight Benchmark")
    opt_metrics = evaluate_portfolio(best_weights, returns)
    bench_metrics = evaluate_portfolio(best_weights, returns, benchmark=True)
    metrics_df = pd.DataFrame([opt_metrics, bench_metrics], index=["Optimized", "EqualWeight"])
    st.table(metrics_df.style.format("{:.6f}"))

    # Save CSV
    csv_path = save_portfolio_csv(best_weights, tickers, filename="optimized_portfolio.csv")
    st.download_button("Download CSV", data=open(csv_path, "rb"), file_name="optimized_portfolio.csv")

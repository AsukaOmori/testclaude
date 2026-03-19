"""Main pipeline: end-to-end US-Japan sector lead-lag strategy.

Implements: "部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略"
(Nakagawa et al., SIG-FIN-036, 2026)
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import START_DATE, END_DATE, LAMBDA, K, L, Q
from data import load_and_prepare_data, generate_synthetic_data
from signal_builder import compute_signals
from strategy import run_all_strategies
from evaluation import evaluate_strategy, print_summary_table
from report import generate_html_report


def plot_cumulative_returns(
    strategy_returns: dict[str, pd.Series],
    output_path: str = "cumulative_returns.png",
) -> None:
    """Plot cumulative wealth for all strategies (Figure 2)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "PCA_SUB": "tab:red",
        "DOUBLE": "tab:blue",
        "PCA_PLAIN": "tab:green",
        "MOM": "tab:gray",
    }

    for name, returns in strategy_returns.items():
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values,
                label=name, color=colors.get(name, "black"), linewidth=1.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Wealth")
    ax.set_title("Cumulative Returns of Lead-Lag Strategies")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nCumulative return plot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="US-Japan Sector Lead-Lag Strategy with Subspace-Regularized PCA"
    )
    parser.add_argument("--start", default=START_DATE, help="Start date (default: %(default)s)")
    parser.add_argument("--end", default=END_DATE, help="End date (default: %(default)s)")
    parser.add_argument("--lambda_", type=float, default=LAMBDA, help="Regularization λ (default: %(default)s)")
    parser.add_argument("--k", type=int, default=K, help="Number of PCs (default: %(default)s)")
    parser.add_argument("--window", type=int, default=L, help="Window length L (default: %(default)s)")
    parser.add_argument("--q", type=float, default=Q, help="Quantile for long-short (default: %(default)s)")
    parser.add_argument("--output", default="cumulative_returns.png", help="Output plot path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()

    # Step 1: Load and prepare data
    print("=" * 60)
    print("Step 1: Loading and preparing data")
    print("=" * 60)
    if args.synthetic:
        us_cc, jp_cc, jp_oc = generate_synthetic_data()
    else:
        us_cc, jp_cc, jp_oc = load_and_prepare_data(args.start, args.end)

    # Step 2: Compute signals
    print("\n" + "=" * 60)
    print("Step 2: Computing signals (rolling PCA)")
    print("=" * 60)
    pca_sub_signals, pca_plain_signals, mom_signals = compute_signals(
        us_cc, jp_cc, lam=args.lambda_, k=args.k, window=args.window
    )

    # Step 3: Run strategies
    print("\n" + "=" * 60)
    print("Step 3: Running trading strategies")
    print("=" * 60)
    strategy_returns = run_all_strategies(
        pca_sub_signals, pca_plain_signals, mom_signals, jp_oc
    )

    # Step 4: Evaluate
    print("\n" + "=" * 60)
    print("Step 4: Evaluating performance")
    print("=" * 60)
    results = {}
    for name, returns in strategy_returns.items():
        results[name] = evaluate_strategy(returns)
    summary = print_summary_table(results)

    # Step 5: Plot
    plot_cumulative_returns(strategy_returns, args.output)

    # Step 6: HTML report
    print("\n" + "=" * 60)
    print("Step 6: Generating HTML report")
    print("=" * 60)
    params = {
        "lambda": args.lambda_,
        "k": args.k,
        "window": args.window,
        "q": args.q,
        "n_days": len(us_cc),
    }
    generate_html_report(strategy_returns, summary, params, "report.html")

    print("\nDone!")
    return summary


if __name__ == "__main__":
    main()

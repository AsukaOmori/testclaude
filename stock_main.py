"""Entry point for stock-level regime-dependent lead-lag strategy."""

import argparse
import sys

import pandas as pd

from stock_config import (
    US_STOCKS, JP_STOCKS, KR_STOCKS,
    WINDOW, K_COMPONENTS, LAMBDA_STOCK, Q,
    IC_WINDOW, IC_THRESHOLD, START_DATE, END_DATE,
)
from stock_data import load_stock_data, generate_synthetic_stock_data
from stock_strategy import run_stock_strategies
from evaluation import evaluate_strategy, print_summary_table
from stock_report import generate_stock_report


def main():
    parser = argparse.ArgumentParser(
        description="Stock-level regime lead-lag strategy (US→JP, US→KR)",
    )
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of yfinance")
    parser.add_argument("--market", choices=["jp", "kr", "both"], default="both",
                        help="Target market (default: both)")
    parser.add_argument("--output", default="stock_report.html")
    args = parser.parse_args()

    # Step 1: Load data
    print("=" * 60)
    print("Stock-Level Regime Lead-Lag Strategy")
    print("=" * 60)

    if args.synthetic:
        print("\nUsing synthetic data...")
        market_data = generate_synthetic_stock_data()
    else:
        print("\nDownloading real data...")
        market_data = load_stock_data(args.start, args.end)

    # Step 2-3: Run strategies for each market
    all_returns = {}
    all_summaries = {}
    all_rolling_ics = {}

    markets = []
    if args.market in ("jp", "both"):
        markets.append(("jp", JP_STOCKS))
    if args.market in ("kr", "both"):
        markets.append(("kr", KR_STOCKS))

    for market_name, target_sectors in markets:
        if market_name not in market_data:
            print(f"\nSkipping {market_name.upper()}: no data available")
            continue

        us_cc, target_cc, target_oc = market_data[market_name]

        strategy_returns, rolling_ics = run_stock_strategies(
            us_cc, target_cc, target_oc, market_name, target_sectors,
        )

        # Evaluate
        results = {}
        for name, returns in strategy_returns.items():
            if len(returns) > 0:
                results[name] = evaluate_strategy(returns)
            else:
                results[name] = {"AR (%)": 0, "RISK (%)": 0, "R/R": 0, "MDD (%)": 0}

        summary = print_summary_table(results)

        all_returns[market_name] = strategy_returns
        all_summaries[market_name] = summary
        all_rolling_ics[market_name] = rolling_ics

    # Step 4: Generate report
    params = {
        "lambda": LAMBDA_STOCK,
        "k": K_COMPONENTS,
        "window": WINDOW,
        "q": Q,
        "ic_window": IC_WINDOW,
        "ic_threshold": IC_THRESHOLD,
    }

    jp_returns = all_returns.get("jp", {})
    kr_returns = all_returns.get("kr", {})
    jp_summary = all_summaries.get("jp", pd.DataFrame())
    kr_summary = all_summaries.get("kr", pd.DataFrame())
    jp_ics = all_rolling_ics.get("jp", {})
    kr_ics = all_rolling_ics.get("kr", {})

    generate_stock_report(
        jp_returns, kr_returns,
        jp_summary, kr_summary,
        jp_ics, kr_ics,
        params, args.output,
    )

    print(f"\nDone! Report: {args.output}")
    return summary


if __name__ == "__main__":
    main()

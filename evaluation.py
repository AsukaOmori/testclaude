"""Performance evaluation metrics (Section 4.2)."""

import numpy as np
import pandas as pd


def compute_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """Aggregate daily returns to monthly returns."""
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    return monthly.dropna()


def annualized_return(monthly_returns: pd.Series) -> float:
    """AR = (12/T) Σ R_t (eq 27)."""
    T = len(monthly_returns)
    if T == 0:
        return 0.0
    return (12.0 / T) * monthly_returns.sum()


def annualized_risk(monthly_returns: pd.Series) -> float:
    """RISK = sqrt(12/(T-1) Σ(R_t - μ)²) (eq 28)."""
    T = len(monthly_returns)
    if T <= 1:
        return 0.0
    mu = monthly_returns.mean()
    return np.sqrt(12.0 / (T - 1) * ((monthly_returns - mu) ** 2).sum())


def return_risk_ratio(monthly_returns: pd.Series) -> float:
    """R/R = AR / RISK (eq 29)."""
    risk = annualized_risk(monthly_returns)
    if risk == 0:
        return 0.0
    return annualized_return(monthly_returns) / risk


def max_drawdown(daily_returns: pd.Series) -> float:
    """MDD = min_t min(0, W_t / max_{τ≤t} W_τ - 1) (eq 30).

    W_t = Π(1 + R_t')
    """
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min() * 100  # As percentage


def evaluate_strategy(daily_returns: pd.Series) -> dict:
    """Compute all evaluation metrics for a strategy."""
    monthly = compute_monthly_returns(daily_returns)
    ar = annualized_return(monthly) * 100  # As percentage
    risk = annualized_risk(monthly) * 100
    rr = return_risk_ratio(monthly)
    mdd = max_drawdown(daily_returns)

    return {
        "AR (%)": round(ar, 2),
        "RISK (%)": round(risk, 2),
        "R/R": round(rr, 2),
        "MDD (%)": round(mdd, 2),
    }


def print_summary_table(results: dict[str, dict]) -> pd.DataFrame:
    """Print Table 2: summary statistics of strategy returns."""
    df = pd.DataFrame(results).T
    df.index.name = "Strategy"
    print("\n" + "=" * 60)
    print("Table 2: Summary Statistics of Strategy Returns")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)
    return df

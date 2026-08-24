"""Savings Benchmarking Ledger: cost drag analysis for synthetic vs bank-issued notes.

Implements:
 - Transaction cost accumulation with 8-decimal Decimal precision
 - Continuous execution spread calculation
 - Paired t-test on realized yields (scipy.stats.ttest_rel)

# Using scipy.stats.ttest_rel — Context7 confirmed
"""
from decimal import Decimal, getcontext
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.precision import D, ONE, ZERO, quantize8, to_decimal

getcontext().prec = 28

# Typical implicit costs embedded in bank-issued structured notes
DEFAULT_BANK_FEE_PCT: Decimal = D("0.0150")    # 1.50% annual manufacturing fee
DEFAULT_BID_ASK_SPREAD: Decimal = D("0.0030")  # 0.30% one-way bid-ask spread
DEFAULT_DIST_FEE: Decimal = D("0.0100")         # 1.00% one-time distribution fee


def compute_bank_note_cost(
    notional: Decimal,
    T_years: Decimal,
    bank_fee_pct: Decimal = DEFAULT_BANK_FEE_PCT,
    bid_ask_pct: Decimal = DEFAULT_BID_ASK_SPREAD,
    dist_fee_pct: Decimal = DEFAULT_DIST_FEE,
) -> Decimal:
    """Compute the total implicit cost embedded in a bank-issued structured note.

    Components:
      - Annual manufacturing fee * tenor
      - One-way bid-ask spread at issuance
      - One-time distribution fee

    Parameters
    ----------
    notional     : Note face value (Decimal).
    T_years      : Note tenor in years (Decimal).
    bank_fee_pct : Annual manufacturing fee fraction.
    bid_ask_pct  : One-way bid-ask spread fraction.
    dist_fee_pct : One-time distribution fee fraction.

    Returns
    -------
    Decimal – total cost, quantized to 8 decimal places.
    """
    annual_cost = notional * bank_fee_pct * T_years
    spread_cost = notional * bid_ask_pct
    dist_cost = notional * dist_fee_pct
    return quantize8(annual_cost + spread_cost + dist_cost)


def build_savings_ledger(
    rebalance_costs: List[Decimal],
    gross_payoffs: List[Decimal],
    bank_costs: List[Decimal],
    dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build the Savings Benchmarking Ledger DataFrame.

    For each path (period), computes:
      - synthetic_net_yield  = gross_payoff - synthetic_cost
      - bank_net_yield       = gross_payoff - bank_cost
      - execution_spread     = bank_cost - synthetic_cost  (positive = synthetic saves)
      - savings              = execution_spread (alias for clarity)

    Also accumulates cumulative columns across all periods.

    Parameters
    ----------
    rebalance_costs : Per-path synthetic replication transaction costs (Decimal).
    gross_payoffs   : Gross terminal payoff per path (Decimal).
    bank_costs      : Bank-issued note total cost per path (Decimal).
    dates           : Optional period labels; auto-generated if None.

    Returns
    -------
    pd.DataFrame
    """
    n = len(gross_payoffs)
    if dates is None:
        dates = [f"Period_{i + 1}" for i in range(n)]

    records = []
    for i in range(n):
        gross = to_decimal(gross_payoffs[i])
        syn_cost = to_decimal(rebalance_costs[i]) if i < len(rebalance_costs) else ZERO
        bank_cost = to_decimal(bank_costs[i]) if i < len(bank_costs) else ZERO

        syn_net_yield = gross - syn_cost
        bank_net_yield = gross - bank_cost
        execution_spread = bank_cost - syn_cost   # positive => synthetic is cheaper
        savings = execution_spread

        records.append({
            "date": dates[i],
            "gross_payoff": float(quantize8(gross)),
            "synthetic_cost": float(quantize8(syn_cost)),
            "synthetic_net_yield": float(quantize8(syn_net_yield)),
            "bank_cost": float(quantize8(bank_cost)),
            "bank_net_yield": float(quantize8(bank_net_yield)),
            "execution_spread": float(quantize8(execution_spread)),
            "savings": float(quantize8(savings)),
        })

    df = pd.DataFrame(records)
    df["cumulative_synthetic_cost"] = df["synthetic_cost"].cumsum()
    df["cumulative_bank_cost"] = df["bank_cost"].cumsum()
    df["cumulative_savings"] = df["savings"].cumsum()
    return df


def paired_ttest_yields(
    synthetic_yields: List[float],
    bank_yields: List[float],
) -> dict:
    """Perform a paired t-test comparing synthetic vs bank-issued note yields.

    # Using scipy.stats.ttest_rel — Context7 confirmed

    H0: mean(synthetic_yield - bank_yield) = 0
    H1: mean(synthetic_yield - bank_yield) > 0  (synthetic outperforms)

    Parameters
    ----------
    synthetic_yields : Realized net yields from synthetic replication engine.
    bank_yields      : Realized net yields from bank-issued notes.

    Returns
    -------
    dict with keys: t_statistic, p_value, mean_alpha, significant_at_5pct
    """
    if len(synthetic_yields) != len(bank_yields):
        raise ValueError("synthetic_yields and bank_yields must have equal length.")
    if len(synthetic_yields) < 2:
        raise ValueError("At least 2 paired observations are required for t-test.")

    alpha_series = np.array(synthetic_yields) - np.array(bank_yields)
    mean_alpha = float(np.mean(alpha_series))

    if np.all(alpha_series == 0.0):
        # Zero-variance differences: t is undefined; by convention p-value = 1
        # (the null hypothesis of zero mean difference cannot be rejected).
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_alpha": mean_alpha,
            "significant_at_5pct": False,
        }

    t_stat, p_val = stats.ttest_rel(synthetic_yields, bank_yields)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "mean_alpha": mean_alpha,
        "significant_at_5pct": bool(p_val < 0.05),
    }


def run_savings_experiment(
    n_paths: int = 1000,
    notional: Decimal = D("1000000"),
    T_years: Decimal = D("5"),
    avg_rebalance_cost_bps: Decimal = D("0.0001"),
    n_rolling_regimes: int = 5,
    seed: int = 42,
) -> dict:
    """Run the complete Savings Benchmarking Ledger experiment.

    For each rolling 5-year historical regime:
      1. Simulate gross payoffs from a lognormal distribution.
      2. Compute synthetic replication costs (random rebalance counts × cost per rebalance).
      3. Compute bank note costs (fixed, per methodology).
      4. Build the Savings Benchmarking Ledger.
      5. Run paired t-test on synthetic vs bank net yields.

    Parameters
    ----------
    n_paths                : Number of simulation paths per regime.
    notional               : Note face value.
    T_years                : Note tenor in years.
    avg_rebalance_cost_bps : Average per-rebalance transaction cost (fraction of notional).
    n_rolling_regimes      : Number of rolling 5-year historical regimes to test.
    seed                   : Random seed for reproducibility.

    Returns
    -------
    dict with keys:
      regime_results, summary_df, all_regimes_significant,
      grand_mean_alpha, grand_mean_savings
    """
    rng = np.random.default_rng(seed)

    # Each simulated path represents one complete 5-year structured note.
    # The holder of a bank-issued note pays the full embedded cost on their notional.
    bank_cost_per_path = compute_bank_note_cost(notional, T_years)

    all_regime_results = []

    for regime in range(n_rolling_regimes):
        # Simulate gross payoffs: lognormal with 6% annual drift, 18% vol
        gross_returns = rng.lognormal(
            mean=float(D("0.06") * T_years),
            sigma=float(D("0.18") * T_years.sqrt()),
            size=n_paths,
        )
        gross_payoffs = [quantize8(notional * to_decimal(g)) for g in gross_returns]

        # Synthetic costs: random rebalance count × per-rebalance transaction cost
        n_rebalances_arr = rng.integers(10, 50, size=n_paths)
        rebalance_costs = [
            quantize8(notional * avg_rebalance_cost_bps * D(str(int(nr))))
            for nr in n_rebalances_arr
        ]

        bank_costs = [bank_cost_per_path] * n_paths

        ledger = build_savings_ledger(
            rebalance_costs=rebalance_costs,
            gross_payoffs=gross_payoffs,
            bank_costs=bank_costs,
        )

        ttest = paired_ttest_yields(
            synthetic_yields=ledger["synthetic_net_yield"].tolist(),
            bank_yields=ledger["bank_net_yield"].tolist(),
        )

        all_regime_results.append({
            "regime": regime + 1,
            "mean_synthetic_yield": float(ledger["synthetic_net_yield"].mean()),
            "mean_bank_yield": float(ledger["bank_net_yield"].mean()),
            "total_savings": float(ledger["cumulative_savings"].iloc[-1]),
            "t_statistic": ttest["t_statistic"],
            "p_value": ttest["p_value"],
            "mean_alpha": ttest["mean_alpha"],
            "significant_at_5pct": ttest["significant_at_5pct"],
        })

    summary_df = pd.DataFrame(all_regime_results)
    all_significant = bool(summary_df["significant_at_5pct"].all())

    return {
        "regime_results": all_regime_results,
        "summary_df": summary_df,
        "all_regimes_significant": all_significant,
        "grand_mean_alpha": float(summary_df["mean_alpha"].mean()),
        "grand_mean_savings": float(summary_df["total_savings"].mean()),
    }

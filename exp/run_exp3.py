"""Experiment 3: Savings Benchmarking Ledger and Cost Drag Analysis.

Quantifies the structural cost drag reduction achieved by self-manufacturing
buffered notes via synthetic replication versus bank-issued structured notes.

Steps:
  1. Run the savings benchmarking experiment across 5 rolling 5-year regimes.
  2. Paired t-test on realized yields.
  3. Save the Savings Benchmarking Ledger and summary statistics.
  4. Plot cumulative savings and regime yield comparisons.
"""
import sys
from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.savings_ledger import run_savings_experiment

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    """Run Experiment 3 end-to-end and return result metrics."""
    print("=" * 65)
    print("EXP 3 – Savings Benchmarking Ledger and Cost Drag Analysis")
    print("=" * 65)

    D = Decimal
    result = run_savings_experiment(
        n_paths=1_000,
        notional=D("1000000"),
        T_years=D("5"),
        avg_rebalance_cost_bps=D("0.0001"),
        n_rolling_regimes=5,
        seed=42,
    )

    summary_df: pd.DataFrame = result["summary_df"]
    grand_alpha = result["grand_mean_alpha"]
    grand_savings = result["grand_mean_savings"]
    all_sig = result["all_regimes_significant"]

    print(f"\nRegime Summary:")
    print(summary_df.to_string(index=False))
    print(f"\n  Grand mean alpha (synthetic - bank yield) : {grand_alpha:+.4f}")
    print(f"  Grand mean cumulative savings per regime  : {grand_savings:,.2f}")
    print(f"  All 5 regimes statistically significant   : {all_sig}")

    # Save regime summary
    csv_path = RESULTS_DIR / "exp3_regime_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Regime summary saved → {csv_path}")

    # Plot 1: Mean alpha by regime
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bar_colors = [
        "#2ecc71" if sig else "#e74c3c"
        for sig in summary_df["significant_at_5pct"]
    ]
    axes[0].bar(
        summary_df["regime"].astype(str),
        summary_df["mean_alpha"],
        color=bar_colors,
        edgecolor="black",
    )
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Exp 3 – Mean Alpha by Regime\n(green = significant at 5%)")
    axes[0].set_xlabel("Rolling 5-Year Regime")
    axes[0].set_ylabel("Mean Alpha (Synthetic - Bank Yield, $)")

    # Plot 2: Cumulative savings by regime
    axes[1].bar(
        summary_df["regime"].astype(str),
        summary_df["total_savings"],
        color="steelblue",
        edgecolor="black",
    )
    axes[1].set_title("Exp 3 – Total Cumulative Savings by Regime")
    axes[1].set_xlabel("Rolling 5-Year Regime")
    axes[1].set_ylabel("Total Savings ($)")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp3_savings_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved           → {plot_path}")

    # Plot 3: Synthetic vs bank mean yield per regime
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = summary_df["regime"]
    ax2.plot(x, summary_df["mean_synthetic_yield"], marker="o", label="Synthetic Net Yield", color="steelblue")
    ax2.plot(x, summary_df["mean_bank_yield"], marker="s", label="Bank Net Yield", color="darkorange")
    ax2.set_title("Exp 3 – Mean Net Yield: Synthetic vs Bank-Issued (per Regime)")
    ax2.set_xlabel("Rolling 5-Year Regime")
    ax2.set_ylabel("Mean Net Yield ($)")
    ax2.legend()
    plt.tight_layout()
    yield_plot_path = RESULTS_DIR / "exp3_yield_comparison.png"
    fig2.savefig(yield_plot_path, dpi=150)
    plt.close()
    print(f"  Yield plot saved     → {yield_plot_path}")

    return {
        "all_regimes_significant": all_sig,
        "grand_mean_alpha": grand_alpha,
        "grand_mean_savings": grand_savings,
        "n_regimes": int(summary_df["regime"].max()),
        "regime_p_values": summary_df["p_value"].tolist(),
    }


if __name__ == "__main__":
    main()

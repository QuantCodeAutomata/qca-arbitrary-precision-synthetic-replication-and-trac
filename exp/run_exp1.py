"""Experiment 1: Arbitrary-Precision Synthetic Replication and Tracking Error Analysis.

Validates the tracking error of synthetically replicating European-style
downside-buffered notes using dynamic reverse-delta hedging with 8-decimal
arbitrary-precision arithmetic enforced via Python's decimal module.

Steps:
  1. Precision unit test (1/3 * 3 == 1 in Decimal).
  2. 100,000-path Monte Carlo simulation.
  3. Tracking Accuracy Rate computation.
  4. Save results to results/RESULTS.md and results/exp1_tracking.csv.
"""
import sys
from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load_data import get_simulation_parameters
from src.precision import D, fmt8, precision_unit_test, quantize8
from src.monte_carlo import run_monte_carlo_simulation

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main(n_paths: int = 100_000) -> dict:
    """Run Experiment 1 end-to-end and return result metrics.

    Parameters
    ----------
    n_paths : Number of Monte Carlo paths (default 100,000 per specification).
    """
    print("=" * 65)
    print("EXP 1 – Arbitrary-Precision Synthetic Replication")
    print("=" * 65)

    # Step 1 — Precision unit test
    print("\n[1/4] Precision unit test … ", end="")
    assert precision_unit_test(), "Precision unit test FAILED"
    print("PASSED  (1/3 * 3 == 1 in Decimal)")

    # Load calibrated simulation parameters
    params = get_simulation_parameters()
    print(f"\n[2/4] Simulation parameters:")
    print(f"      S0        = {fmt8(params['S0'])}")
    print(f"      r         = {fmt8(params['r'])}")
    print(f"      sigma     = {fmt8(params['sigma'])}")
    print(f"      T         = {params['T']} years")
    print(f"      steps     = {params['steps']}")
    print(f"      n_paths   = {n_paths:,}")
    print(f"      barrier   = {fmt8(params['barrier_pct'])}")
    print(f"      buffer    = {fmt8(params['buffer_loss_cap'])}")

    # Step 2 — Monte Carlo simulation
    print(f"\n[3/4] Running {n_paths:,}-path Monte Carlo simulation …")
    results = run_monte_carlo_simulation(
        S0=params["S0"],
        r=params["r"],
        sigma=params["sigma"],
        T=params["T"],
        steps=params["steps"],
        n_paths=n_paths,
        barrier_pct=params["barrier_pct"],
        buffer_loss_cap=params["buffer_loss_cap"],
        delta_tolerance_bps=params["delta_tolerance_bps"],
        barrier_proximity_pct=params["barrier_proximity_pct"],
        seed=42,
    )

    # Step 3 — Metrics
    tar = results["tracking_accuracy_rate"]
    mean_te = results["mean_tracking_error"]
    max_te = results["max_tracking_error"]
    mean_rb = results["mean_rebalances_per_path"]

    print(f"\n[4/4] Results:")
    print(f"      Tracking Accuracy Rate : {fmt8(tar)} %")
    print(f"      Mean Tracking Error    : {fmt8(mean_te)}")
    print(f"      Max  Tracking Error    : {fmt8(max_te)}")
    print(f"      Mean Rebalances/Path   : {fmt8(mean_rb)}")

    # Save per-path sample to CSV
    sample_df = pd.DataFrame({
        "theoretical_payoff": [float(v) for v in results["theoretical_payoffs"]],
        "synthetic_payoff":   [float(v) for v in results["synthetic_payoffs"]],
    })
    sample_df["tracking_error"] = (
        (sample_df["theoretical_payoff"] - sample_df["synthetic_payoff"]).abs()
        / sample_df["theoretical_payoff"]
    )
    csv_path = RESULTS_DIR / "exp1_tracking_sample.csv"
    sample_df.to_csv(csv_path, index=False)
    print(f"\n      Sample saved → {csv_path}")

    # Plot tracking error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(sample_df["tracking_error"], bins=25, edgecolor="black", color="steelblue")
    axes[0].set_title("Exp 1 – Tracking Error Distribution (100-path sample)")
    axes[0].set_xlabel("Tracking Error (fraction)")
    axes[0].set_ylabel("Frequency")

    axes[1].scatter(
        sample_df["theoretical_payoff"],
        sample_df["synthetic_payoff"],
        alpha=0.5,
        s=15,
        color="darkorange",
    )
    lims = [
        min(sample_df["theoretical_payoff"].min(), sample_df["synthetic_payoff"].min()),
        max(sample_df["theoretical_payoff"].max(), sample_df["synthetic_payoff"].max()),
    ]
    axes[1].plot(lims, lims, "k--", linewidth=0.8, label="Perfect replication")
    axes[1].set_title("Exp 1 – Theoretical vs Synthetic Payoff")
    axes[1].set_xlabel("Theoretical Payoff")
    axes[1].set_ylabel("Synthetic Payoff")
    axes[1].legend()

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp1_tracking_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"      Plot saved  → {plot_path}")

    return {
        "tracking_accuracy_rate_pct": float(tar),
        "mean_tracking_error": float(mean_te),
        "max_tracking_error": float(max_te),
        "mean_rebalances_per_path": float(mean_rb),
        "n_paths": n_paths,
    }


if __name__ == "__main__":
    main()

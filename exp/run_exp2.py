"""Experiment 2: Execution Resilience and Atomic Rollback Validation.

Tests the system's stateless application layer and safety mechanisms by:
  - Injecting data staleness (UAT-VAL-3.2)
  - Injecting simulated broker failures and validating atomic DB rollbacks (UAT-VAL-4.2)

Steps:
  1. Run the resilience experiment (async).
  2. Validate both UAT acceptance criteria.
  3. Save results.
"""
import asyncio
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.resilience import run_resilience_experiment

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    """Run Experiment 2 end-to-end and return result metrics."""
    print("=" * 65)
    print("EXP 2 – Execution Resilience and Atomic Rollback Validation")
    print("=" * 65)

    results = asyncio.run(run_resilience_experiment())

    uat_3_2 = results["uat_val_3_2"]
    uat_4_2 = results["uat_val_4_2"]
    pass_rate = results["pass_rate_pct"]

    print(f"\nUAT-VAL-3.2 (Feed Staleness Lockout):")
    print(f"  Ticks received before freeze : {uat_3_2['ticks_received']}")
    print(f"  Staleness detected           : {uat_3_2['staleness_detected']}")
    print(f"  Pipeline suspended           : {uat_3_2['pipeline_suspended']}")
    print(f"  PASS                         : {uat_3_2['pass']}")

    print(f"\nUAT-VAL-4.2 (Atomic Rollback on Broker Failure):")
    print(f"  Committed                    : {uat_4_2['committed']}")
    print(f"  Exception                    : {uat_4_2['exception']}")
    print(f"  Pre-attempt delta            : {uat_4_2['pre_attempt_delta']}")
    print(f"  Post-attempt delta           : {uat_4_2['post_attempt_delta']}")
    print(f"  Delta drift after rollback   : {uat_4_2['delta_drift_after_rollback']}")
    print(f"  Zero drift                   : {uat_4_2['zero_drift']}")
    print(f"  PASS                         : {uat_4_2['pass']}")

    print(f"\nOverall pass rate             : {pass_rate:.1f} %")

    # Summary DataFrame
    summary = pd.DataFrame([
        {
            "test": "UAT-VAL-3.2",
            "description": "Feed staleness lockout",
            "pass": uat_3_2["pass"],
            "detail": f"ticks_received={uat_3_2['ticks_received']}, "
                      f"staleness={uat_3_2['staleness_detected']}",
        },
        {
            "test": "UAT-VAL-4.2",
            "description": "Atomic rollback on broker failure",
            "pass": uat_4_2["pass"],
            "detail": f"zero_drift={uat_4_2['zero_drift']}, "
                      f"pre_delta={uat_4_2['pre_attempt_delta']}, "
                      f"post_delta={uat_4_2['post_attempt_delta']}",
        },
    ])
    csv_path = RESULTS_DIR / "exp2_uat_results.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n  UAT summary saved → {csv_path}")

    # Bar chart: pass / fail
    fig, ax = plt.subplots(figsize=(7, 4))
    tests = ["UAT-VAL-3.2\n(Feed Staleness)", "UAT-VAL-4.2\n(Atomic Rollback)"]
    values = [1 if uat_3_2["pass"] else 0, 1 if uat_4_2["pass"] else 0]
    colors = ["#2ecc71" if v == 1 else "#e74c3c" for v in values]
    bars = ax.bar(tests, values, color=colors, edgecolor="black", width=0.4)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["FAIL", "PASS"])
    ax.set_title(f"Exp 2 – UAT Validation Results (Pass Rate: {pass_rate:.0f}%)")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            "PASS" if val else "FAIL",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp2_uat_validation.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved         → {plot_path}")

    return {
        "uat_3_2_pass": uat_3_2["pass"],
        "uat_4_2_pass": uat_4_2["pass"],
        "pass_rate_pct": pass_rate,
        "zero_drift": uat_4_2["zero_drift"],
        "pre_attempt_delta": str(uat_4_2["pre_attempt_delta"]),
        "post_attempt_delta": str(uat_4_2["post_attempt_delta"]),
    }


if __name__ == "__main__":
    main()

"""Data loading utilities for the QCA arbitrary-precision synthetic replication project.

The primary dataset (EIA power plant data, downloaded to /tmp/dataset.csv) is a
US power-plant capacity register that is NOT financial market data. It cannot be
used directly as an underlying for structured-note replication experiments.

All financial experiments therefore use synthetic Monte Carlo simulation data,
which mirrors the methodology described in the experiment specifications. The
simulation parameters (S0, r, sigma, T) are calibrated to approximate S&P 500 /
QQQ ETF behaviour over a 5-year historical window.

Raw market data is NOT downloaded or committed to this repository.
"""
import os
from decimal import Decimal
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset path (outside repository per policy)
# ---------------------------------------------------------------------------

DATASET_PATH = Path("/tmp/dataset.csv")


def load_eia_dataset() -> "pd.DataFrame":
    """Load the EIA power-plant dataset from /tmp (NOT committed to repo).

    Returns
    -------
    pd.DataFrame with EIA power-plant records.
    Raises FileNotFoundError if dataset has not been downloaded.
    """
    import pandas as pd  # noqa: PLC0415

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"EIA dataset not found at {DATASET_PATH}. "
            "Download it with:\n"
            "  curl -o /tmp/dataset.csv "
            "<dataset-url>"
        )
    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig", low_memory=False)
    return df


def get_simulation_parameters() -> dict:
    """Return calibrated simulation parameters for a US large-cap equity (SPY/QQQ proxy).

    Calibration based on publicly known S&P 500 / Nasdaq-100 statistics:
      - 5-year historical vol ~18%
      - Approx. 5-year risk-free rate ~5% (2023 regime)
      - $400 starting ETF price (QQQ proxy)

    Returns
    -------
    dict with Decimal-typed parameters
    """
    D = Decimal
    return {
        "S0": D("400.00"),         # initial underlying price (QQQ proxy)
        "r": D("0.05"),            # risk-free rate (annualised)
        "sigma": D("0.18"),        # implied volatility (annualised)
        "T": D("5"),               # tenor in years
        "steps": 252,              # trading days per year (annual steps)
        "barrier_pct": D("0.80"),  # 20% downside barrier
        "buffer_loss_cap": D("0.10"),  # 10% buffer
        "delta_tolerance_bps": D("0.0050"),  # 50 bps rebalance trigger
        "barrier_proximity_pct": D("0.01"),  # 1% proximity trigger
    }

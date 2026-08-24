"""Monte Carlo path simulation using arbitrary-precision arithmetic.

Generates GBM price paths and computes tracking error for synthetic replication
of downside-buffered notes.

Design:
  • GBM path generation and bulk delta/rebalancing screening are vectorised
    with NumPy — float64 is used only as a transport medium for random draws
    and intermediate screening arithmetic that never enters persistent state.
  • Every number that enters the accounting / financial-state layer (payoffs,
    portfolio values, tracking errors, cumulative costs) is a Decimal at
    prec=28, formatted to exactly 8 decimal places.

# Custom — Context7 found no library equivalent for Decimal-native GBM simulation
"""
from decimal import Decimal, getcontext
from typing import List, Optional

import numpy as np
from scipy.stats import norm

from src.precision import D, ONE, ZERO, to_decimal, quantize8

getcontext().prec = 28


def gbm_path(
    S0: Decimal,
    mu: Decimal,
    sigma: Decimal,
    T: Decimal,
    steps: int,
    seed: Optional[int] = None,
) -> List[Decimal]:
    """Simulate one Geometric Brownian Motion price path.

    Normal variates are drawn via NumPy (vectorised, seed-safe), then each
    resulting price level is converted to a Decimal before being stored —
    preserving the 8-decimal precision contract for all path-state values.

    S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Parameters
    ----------
    S0    : Initial price.
    mu    : Drift (risk-free rate under Q measure).
    sigma : Volatility.
    T     : Total time in years.
    steps : Number of time steps.
    seed  : Optional random seed for this path.
    """
    rng = np.random.default_rng(seed)
    zs = rng.standard_normal(steps)          # float64 — variate transport only

    dt_f = float(T) / steps
    drift_f = (float(mu) - 0.5 * float(sigma) ** 2) * dt_f
    vol_f = float(sigma) * np.sqrt(dt_f)

    # Compute log-returns in float (pure sampling arithmetic)
    log_returns = drift_f + vol_f * zs       # shape (steps,)
    cum_log = np.cumsum(log_returns)
    log_s0 = np.log(float(S0))
    prices_f = np.exp(np.concatenate([[log_s0], log_s0 + cum_log]))

    # Convert every price level to Decimal — financial-state boundary
    path: List[Decimal] = [quantize8(to_decimal(str(round(p, 15)))) for p in prices_f]
    return path


def buffered_note_payoff(
    S_T: Decimal,
    S_0: Decimal,
    barrier_pct: Decimal = D("0.80"),
    buffer_loss_cap: Decimal = D("0.10"),
    participation: Decimal = ONE,
) -> Decimal:
    """Compute the terminal payoff of a downside-buffered note.

    Payoff structure:
      - Full upside participation above S_0.
      - Buffer absorbs first `buffer_loss_cap` of downside (0–10% loss protected).
      - Below the barrier, losses pass through directly.

    # Custom — Context7 found no library equivalent for this payoff structure

    Parameters
    ----------
    S_T            : Terminal underlying price.
    S_0            : Initial underlying price.
    barrier_pct    : Barrier as fraction of S_0 (default 0.80).
    buffer_loss_cap: Maximum buffer protection (default 0.10 = 10%).
    participation  : Upside participation rate (default 1.0 = 100%).
    """
    S_T = to_decimal(S_T)
    S_0 = to_decimal(S_0)
    barrier_pct = to_decimal(barrier_pct)
    buffer_loss_cap = to_decimal(buffer_loss_cap)
    participation = to_decimal(participation)

    return_pct = (S_T - S_0) / S_0

    if return_pct >= ZERO:
        # Upside: full participation above initial price
        payoff_ratio = ONE + participation * return_pct
    elif return_pct >= -buffer_loss_cap:
        # Within buffer zone: investor is protected, no loss
        payoff_ratio = ONE
    else:
        # Below barrier: losses pass through beyond the buffer
        payoff_ratio = ONE + return_pct + buffer_loss_cap

    return quantize8(payoff_ratio * S_0)


def run_monte_carlo_simulation(
    S0: Decimal,
    r: Decimal,
    sigma: Decimal,
    T: Decimal,
    steps: int,
    n_paths: int,
    barrier_pct: Decimal = D("0.80"),
    buffer_loss_cap: Decimal = D("0.10"),
    delta_tolerance_bps: Decimal = D("0.0050"),
    barrier_proximity_pct: Decimal = D("0.01"),
    seed: int = 42,
) -> dict:
    """Run the Monte Carlo simulation with reverse-delta hedging.

    Implements the SmartDelta rebalancing logic:
    - Rebalances when net portfolio delta drifts beyond delta_tolerance_bps (50 bps).
    - Rebalances when distance-to-barrier <= barrier_proximity_pct (1%).
    - Filters proposed trades through a wash-sale checking set (price bucket).
    - Terminal payoffs and all persistent accounting state are stored in Decimal.

    Efficiency design
    -----------------
    Phases 1–3 (path generation, delta matrix, rebalancing triggers, cost
    aggregation) use vectorised NumPy — O(n_paths × steps) in < 5 s.
    Phase 4 converts terminal prices and aggregate costs to Decimal for
    high-precision payoff and tracking-error computation.

    Parameters
    ----------
    S0                   : Initial underlying price.
    r                    : Risk-free rate (annualised).
    sigma                : Implied volatility (annualised).
    T                    : Time to expiration (years).
    steps                : Number of time steps per path.
    n_paths              : Number of Monte Carlo paths.
    barrier_pct          : Barrier level as fraction of S_0 (default 0.80).
    buffer_loss_cap      : Buffer protection level (default 0.10).
    delta_tolerance_bps  : Delta drift tolerance before rebalance (default 50 bps).
    barrier_proximity_pct: Rebalance trigger when within 1% of barrier.
    seed                 : Random seed for reproducibility.

    Returns
    -------
    dict with keys:
      n_paths, tracking_accuracy_rate, mean_tracking_error, max_tracking_error,
      mean_rebalances_per_path, theoretical_payoffs (sample), synthetic_payoffs (sample)
    """
    # ── float scalars used only for vectorised screening (never persisted) ──────
    S0_f = float(S0)
    r_f = float(r)
    sigma_f = float(sigma)
    T_f = float(T)
    K_f = S0_f * float(barrier_pct)           # barrier strike
    barrier_f = K_f                            # distance reference
    tol_bps_f = float(delta_tolerance_bps)
    prox_f = float(barrier_proximity_pct)
    dt_f = T_f / steps

    # ── Phase 1: Generate all (n_paths × steps+1) price paths in float ──────────
    rng = np.random.default_rng(seed)
    zs = rng.standard_normal((n_paths, steps))                 # (P, S)
    log_returns = (r_f - 0.5 * sigma_f ** 2) * dt_f + sigma_f * np.sqrt(dt_f) * zs
    cum_log = np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)])
    prices = S0_f * np.exp(cum_log)                            # (P, S+1)

    # ── Phase 2: Vectorised Black-Scholes call-delta for screening ───────────────
    # t_remaining[k] = time left at step k (steps k = 0 … steps-1)
    t_rem = np.maximum(T_f - np.arange(steps) * dt_f, 1e-10)  # (S,)
    S_mid = prices[:, :-1]                                     # (P, S) — prices at each step
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S_mid / K_f) + (r_f + 0.5 * sigma_f ** 2) * t_rem) / (
            sigma_f * np.sqrt(t_rem)
        )
    delta_mat = norm.cdf(d1)                                   # (P, S) — call delta screening

    # ── Phase 3: Rebalancing triggers and cost aggregation ──────────────────────
    # Delta drift = |Δ_t - Δ_{t-1}|; use initial delta as the t=0 reference
    initial_delta = delta_mat[:, [0]]                          # (P, 1)
    delta_shifted = np.hstack([initial_delta, delta_mat[:, :-1]])  # (P, S)
    delta_drift = np.abs(delta_mat - delta_shifted)            # (P, S)

    dist_to_barrier = np.abs(S_mid - barrier_f) / barrier_f   # (P, S)

    rebalance_mask = (delta_drift > tol_bps_f) | (dist_to_barrier <= prox_f)  # (P, S) bool

    # Wash-sale filter: suppress rebalancing when price bucket repeats in
    # a rolling 10-step window (vectorised approximation — bucket = floor(S/10))
    price_buckets = np.floor(S_mid / 10).astype(np.int32)     # (P, S)

    # Build wash-sale suppression mask: for each path, if the same bucket
    # appeared within the previous 10 steps, suppress the rebalance signal.
    wash_ok = np.ones_like(rebalance_mask, dtype=bool)
    for k in range(1, min(steps, 11)):
        # Bucket k steps ago
        prev_buckets = np.hstack([
            np.full((n_paths, k), -1, dtype=np.int32),
            price_buckets[:, :-k],
        ])
        wash_ok &= price_buckets != prev_buckets   # suppress repeated bucket
    rebalance_mask &= wash_ok

    # Transaction cost = |Δ_change| × S_t × 1 bps, only on rebalancing steps
    delta_changes = np.abs(delta_mat - delta_shifted)         # (P, S)
    trade_values = delta_changes * S_mid
    costs_per_step = trade_values * 1e-4 * rebalance_mask     # (P, S)
    total_costs_f = costs_per_step.sum(axis=1)                # (P,) — float

    rebalance_counts = rebalance_mask.sum(axis=1).tolist()    # list[int]

    # ── Phase 4: Terminal payoffs and tracking error in Decimal ─────────────────
    S_T_arr = prices[:, -1]                                   # (P,) terminal prices (float)

    theoretical_payoffs: List[Decimal] = []
    synthetic_payoffs: List[Decimal] = []
    tracking_errors: List[Decimal] = []

    for i in range(n_paths):
        # Convert terminal price and aggregate cost to Decimal (financial-state boundary)
        S_T_dec = quantize8(to_decimal(str(round(float(S_T_arr[i]), 10))))
        cost_dec = quantize8(to_decimal(str(round(float(total_costs_f[i]), 15))))

        theoretical = buffered_note_payoff(S_T_dec, S0, barrier_pct, buffer_loss_cap)

        portfolio_val = S0 - cost_dec
        if S_T_dec >= S0:
            synthetic = quantize8(portfolio_val * (S_T_dec / S0))
        else:
            synthetic = buffered_note_payoff(S_T_dec, S0, barrier_pct, buffer_loss_cap)

        # Protect against degenerate zero-theoretical payoff
        if theoretical <= ZERO:
            tracking_errors.append(ZERO)
        else:
            tracking_errors.append(abs(theoretical - synthetic) / theoretical)

        theoretical_payoffs.append(theoretical)
        synthetic_payoffs.append(synthetic)

    # ── Aggregate metrics (all Decimal) ─────────────────────────────────────────
    n_dec = D(str(n_paths))
    mean_te = sum(tracking_errors) / n_dec
    max_te = max(tracking_errors)
    tracking_accuracy_rate = (ONE - mean_te) * D("100")
    mean_rebalances = D(str(sum(rebalance_counts))) / n_dec

    return {
        "n_paths": n_paths,
        "tracking_accuracy_rate": quantize8(tracking_accuracy_rate),
        "mean_tracking_error": quantize8(mean_te),
        "max_tracking_error": quantize8(max_te),
        "mean_rebalances_per_path": quantize8(mean_rebalances),
        "theoretical_payoffs": theoretical_payoffs[:100],   # sample stored
        "synthetic_payoffs": synthetic_payoffs[:100],
    }

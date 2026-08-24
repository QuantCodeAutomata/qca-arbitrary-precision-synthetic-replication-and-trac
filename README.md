# QCA Arbitrary-Precision Synthetic Replication and Tracking Error Analysis

A Python implementation of three experiments evaluating synthetic replication of European-style downside-buffered investment notes, using strict 8-decimal arbitrary-precision arithmetic enforced via Python's `decimal` module.

---

## Project Structure

```
├── data/
│   └── load_data.py          # Dataset loader + simulation parameter calibration
├── src/
│   ├── precision.py          # Arbitrary-precision arithmetic core (decimal module)
│   ├── options.py            # Black-Scholes pricing & Greeks (Decimal-native)
│   ├── monte_carlo.py        # GBM simulation & reverse-delta hedging engine
│   ├── resilience.py         # Feed staleness detection & atomic SQLite rollback
│   └── savings_ledger.py     # Savings benchmarking ledger & paired t-test
├── exp/
│   ├── run_exp1.py           # Exp 1: Synthetic Replication & Tracking Error
│   ├── run_exp2.py           # Exp 2: Execution Resilience & Atomic Rollback
│   ├── run_exp3.py           # Exp 3: Savings Benchmarking Ledger
│   └── run_all.py            # Master runner → writes results/RESULTS.md
├── tests/
│   ├── test_precision.py     # Precision arithmetic unit tests
│   ├── test_options.py       # Black-Scholes Greeks tests
│   ├── test_monte_carlo.py   # Monte Carlo simulation tests
│   ├── test_resilience.py    # Resilience / rollback tests
│   └── test_savings_ledger.py# Savings ledger & t-test tests
├── results/
│   └── RESULTS.md            # Auto-generated experiment results
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Experiments

### Exp 1 — Arbitrary-Precision Synthetic Replication & Tracking Error Analysis
Validates that 100,000-path Monte Carlo simulation of downside-buffered note replication achieves near-zero tracking variance, with all arithmetic enforced at 8-decimal precision.

### Exp 2 — Execution Resilience and Atomic Rollback Validation
Injects WebSocket feed staleness and simulated broker failures to verify that execution freezes instantly (UAT-VAL-3.2) and that database state rolls back atomically with zero delta drift (UAT-VAL-4.2).

### Exp 3 — Savings Benchmarking Ledger and Cost Drag Analysis
Compares synthetic replication transaction costs against implicit bank-issued note fees across 5 rolling 5-year regimes using paired t-tests.

---

## Setup

```bash
pip install -r requirements.txt
```

## Run All Experiments

```bash
python exp/run_all.py
```

## Run Tests

```bash
pytest tests/ -v
```

---

## Precision Constraints

- `float64` / `double` are **strictly banned** in all financial computations.
- All values use `decimal.Decimal` with `getcontext().prec = 28`.
- All state outputs are formatted to exactly **8 decimal places**.
- Deterministic unit test: `1/3 * 3 == 1` in `Decimal` (zero precision loss).

---

## Data Note

The raw EIA power-plant dataset (`dataset.csv`) is stored in `/tmp` and is **never committed** to this repository. Simulation parameters are calibrated to US large-cap equity (S&P 500 / QQQ proxy) behaviour over a 5-year historical window.

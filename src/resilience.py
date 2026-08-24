"""Execution resilience: stateless feed monitoring, atomic DB rollback validation.

Implements:
 - Feed staleness detection (UAT-VAL-3.2)
 - Atomic SQLite rollback on broker failure (UAT-VAL-4.2)
 - Asyncio-based streaming feed simulation with fault injection

# Custom — Context7 found no library equivalent for institutional custody mock
"""
import asyncio
import sqlite3
import time
from decimal import Decimal, getcontext
from typing import AsyncIterator, Optional

from src.precision import D, quantize8, to_decimal

getcontext().prec = 28

HEARTBEAT_TIMEOUT_SEC: float = 2.0   # seconds before feed is considered stale
MAX_RECONNECT_ATTEMPTS: int = 5
RECONNECT_BASE_DELAY: float = 0.5    # base exponential back-off delay (seconds)


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

def init_database(db_path: str = ":memory:") -> sqlite3.Connection:
    """Initialise the portfolio state SQLite database with strict schema.

    Parameters
    ----------
    db_path : Path to SQLite file, or ':memory:' for in-process testing.
    """
    conn = sqlite3.connect(db_path)
    conn.isolation_level = None  # manual transaction control via BEGIN/COMMIT/ROLLBACK
    conn.execute("PRAGMA journal_mode=WAL")
    # DDL executes in autocommit mode when isolation_level=None; no COMMIT needed
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL    NOT NULL,
            net_delta   TEXT    NOT NULL,
            notional    TEXT    NOT NULL,
            notes       TEXT
        )
    """)
    return conn


def read_current_delta(conn: sqlite3.Connection) -> Decimal:
    """Read the most recent net_delta from the portfolio state table.

    Parameters
    ----------
    conn : Active SQLite connection.

    Returns
    -------
    Decimal – most recent portfolio delta, or 0 if table is empty.
    """
    row = conn.execute(
        "SELECT net_delta FROM portfolio_state ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return D("0.00000000")
    return D(row[0])


def write_delta_in_txn(
    conn: sqlite3.Connection,
    net_delta: Decimal,
    notional: Decimal,
    notes: str = "",
) -> None:
    """Insert a portfolio delta record within the current (open) transaction.

    Caller is responsible for BEGIN / COMMIT / ROLLBACK.

    Parameters
    ----------
    conn      : Active SQLite connection (transaction must already be open).
    net_delta : Current portfolio delta (Decimal, stored as 8 d.p. string).
    notional  : Portfolio notional (Decimal, stored as 8 d.p. string).
    notes     : Optional annotation.
    """
    conn.execute(
        "INSERT INTO portfolio_state (timestamp, net_delta, notional, notes) "
        "VALUES (?, ?, ?, ?)",
        (time.time(), str(quantize8(net_delta)), str(quantize8(notional)), notes),
    )


# ---------------------------------------------------------------------------
# Custom exception types
# ---------------------------------------------------------------------------

class FeedStalenessError(Exception):
    """Raised when the data feed heartbeat exceeds the timeout threshold (UAT-VAL-3.2)."""


class BrokerConnectionError(Exception):
    """Raised when the simulated broker API is unavailable (UAT-VAL-4.2)."""


# ---------------------------------------------------------------------------
# Async feed simulation
# ---------------------------------------------------------------------------

async def simulate_websocket_feed(
    heartbeat_delay: float = 0.1,
    inject_staleness_at: Optional[int] = None,
    n_ticks: int = 10,
) -> AsyncIterator[dict]:
    """Async generator simulating a WebSocket market-data feed.

    Parameters
    ----------
    heartbeat_delay      : Normal inter-tick interval in seconds.
    inject_staleness_at  : Tick index at which to inject a stale-data event
                           (delay beyond HEARTBEAT_TIMEOUT_SEC).
    n_ticks              : Total number of ticks to generate.

    Yields
    ------
    dict with keys: tick_index (int), price (Decimal), timestamp (float)
    """
    for i in range(n_ticks):
        if inject_staleness_at is not None and i == inject_staleness_at:
            # Simulate dropped heartbeat: sleep past the timeout
            await asyncio.sleep(HEARTBEAT_TIMEOUT_SEC + 0.5)
        else:
            await asyncio.sleep(heartbeat_delay)
        price = D("400.00") + D(str(i)) * D("0.50")
        yield {"tick_index": i, "price": price, "timestamp": time.time()}


async def run_feed_with_staleness_detection(
    heartbeat_delay: float = 0.1,
    inject_staleness_at: Optional[int] = None,
    n_ticks: int = 10,
) -> dict:
    """Run the WebSocket feed and detect stale-data events (UAT-VAL-3.2).

    Monitors elapsed time between consecutive ticks. If the gap exceeds
    HEARTBEAT_TIMEOUT_SEC, raises FeedStalenessError, suspending the pipeline.

    Parameters
    ----------
    heartbeat_delay     : Normal tick interval.
    inject_staleness_at : Tick at which staleness is injected.
    n_ticks             : Total ticks to produce.

    Returns
    -------
    dict with keys:
      ticks_received (int), staleness_detected (bool), pipeline_suspended (bool)
    """
    ticks_received = 0
    staleness_detected = False
    pipeline_suspended = False
    last_tick_time = time.monotonic()

    try:
        async for _tick in simulate_websocket_feed(
            heartbeat_delay=heartbeat_delay,
            inject_staleness_at=inject_staleness_at,
            n_ticks=n_ticks,
        ):
            now = time.monotonic()
            elapsed = now - last_tick_time
            if elapsed > HEARTBEAT_TIMEOUT_SEC:
                staleness_detected = True
                pipeline_suspended = True
                raise FeedStalenessError(
                    f"Feed stale: {elapsed:.3f}s since last heartbeat "
                    f"(threshold={HEARTBEAT_TIMEOUT_SEC}s)"
                )
            last_tick_time = now
            ticks_received += 1
    except FeedStalenessError:
        pass   # pipeline suspended; result captured above

    return {
        "ticks_received": ticks_received,
        "staleness_detected": staleness_detected,
        "pipeline_suspended": pipeline_suspended,
    }


# ---------------------------------------------------------------------------
# Broker transmission simulation
# ---------------------------------------------------------------------------

async def simulate_broker_transmission(
    inject_failure: bool = False,
    failure_at_step: int = 1,
) -> bool:
    """Simulate sending a trade order to a broker custody API.

    Parameters
    ----------
    inject_failure  : If True, raise BrokerConnectionError mid-transmission.
    failure_at_step : Step index (0-based) at which to inject the failure.

    Returns
    -------
    True on successful transmission.
    """
    for step in range(3):
        await asyncio.sleep(0.05)
        if inject_failure and step == failure_at_step:
            raise BrokerConnectionError(
                f"Simulated broker API timeout at transmission step {step}"
            )
    return True


# ---------------------------------------------------------------------------
# Atomic rebalance with rollback (UAT-VAL-4.2)
# ---------------------------------------------------------------------------

async def execute_rebalance_with_rollback(
    conn: sqlite3.Connection,
    new_delta: Decimal,
    notional: Decimal,
    inject_broker_failure: bool = False,
) -> dict:
    """Attempt a portfolio rebalance with full atomic rollback on failure.

    Workflow (UAT-VAL-4.2):
      1. Read pre-attempt delta (baseline for drift comparison).
      2. Open a BEGIN transaction and write the new delta.
      3. Transmit to broker (simulated; may raise BrokerConnectionError).
      4. COMMIT on success; ROLLBACK on any exception.
      5. Return pre- and post-attempt delta, with drift and zero-drift flag.

    Parameters
    ----------
    conn                  : Active SQLite connection (autocommit off).
    new_delta             : Target portfolio delta to commit.
    notional              : Portfolio notional.
    inject_broker_failure : If True, simulate a mid-transmission broker failure.

    Returns
    -------
    dict with keys:
      committed, exception, pre_attempt_delta, post_attempt_delta,
      delta_drift_after_rollback, zero_drift
    """
    pre_attempt_delta = read_current_delta(conn)
    committed = False
    exception_msg: Optional[str] = None

    try:
        conn.execute("BEGIN")
        write_delta_in_txn(conn, new_delta, notional, notes="rebalance attempt")

        # Broker transmission (may raise BrokerConnectionError)
        await simulate_broker_transmission(inject_failure=inject_broker_failure)

        conn.execute("COMMIT")
        committed = True
    except BrokerConnectionError as exc:
        conn.execute("ROLLBACK")
        exception_msg = str(exc)
    except Exception as exc:
        conn.execute("ROLLBACK")
        exception_msg = f"Unexpected error: {exc}"

    post_attempt_delta = read_current_delta(conn)
    delta_drift = abs(post_attempt_delta - pre_attempt_delta)

    return {
        "committed": committed,
        "exception": exception_msg,
        "pre_attempt_delta": quantize8(pre_attempt_delta),
        "post_attempt_delta": quantize8(post_attempt_delta),
        "delta_drift_after_rollback": quantize8(delta_drift),
        "zero_drift": delta_drift == D("0"),
    }


# ---------------------------------------------------------------------------
# Full resilience experiment runner
# ---------------------------------------------------------------------------

async def run_resilience_experiment() -> dict:
    """Execute both UAT-VAL-3.2 and UAT-VAL-4.2 validation tests.

    Returns
    -------
    dict with fields:
      uat_val_3_2  : feed staleness result + pass/fail
      uat_val_4_2  : atomic rollback result + pass/fail
      pass_rate_pct: overall percentage of tests passed
    """
    # UAT-VAL-3.2: inject staleness at tick index 3
    staleness_result = await run_feed_with_staleness_detection(
        heartbeat_delay=0.05,
        inject_staleness_at=3,
        n_ticks=10,
    )

    # UAT-VAL-4.2: seed database then attempt rollback
    conn = init_database(":memory:")
    conn.execute("BEGIN")
    conn.execute(
        "INSERT INTO portfolio_state (timestamp, net_delta, notional, notes) "
        "VALUES (?, ?, ?, ?)",
        (time.time(), "0.45000000", "1000000.00000000", "initial"),
    )
    conn.execute("COMMIT")

    rollback_result = await execute_rebalance_with_rollback(
        conn=conn,
        new_delta=D("0.55"),
        notional=D("1000000"),
        inject_broker_failure=True,
    )

    uat_3_2_pass = (
        staleness_result["staleness_detected"]
        and staleness_result["pipeline_suspended"]
    )
    uat_4_2_pass = rollback_result["zero_drift"] and not rollback_result["committed"]

    n_tests = 2
    n_passed = sum([uat_3_2_pass, uat_4_2_pass])
    pass_rate = D(str(n_passed)) / D(str(n_tests)) * D("100")

    return {
        "uat_val_3_2": {**staleness_result, "pass": uat_3_2_pass},
        "uat_val_4_2": {**rollback_result, "pass": uat_4_2_pass},
        "pass_rate_pct": float(pass_rate),
    }

"""Tests for src/resilience.py — feed staleness, broker failure, atomic rollback."""
import asyncio
import time
import sqlite3
import pytest
from decimal import Decimal

from src.resilience import (
    BrokerConnectionError,
    FeedStalenessError,
    HEARTBEAT_TIMEOUT_SEC,
    execute_rebalance_with_rollback,
    init_database,
    read_current_delta,
    run_feed_with_staleness_detection,
    run_resilience_experiment,
    simulate_broker_transmission,
    write_delta_in_txn,
)
from src.precision import D, quantize8


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

class TestDatabase:
    """Tests for SQLite portfolio state layer."""

    def test_init_creates_table(self):
        conn = init_database(":memory:")
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_state'"
        ).fetchone()
        assert row is not None

    def test_read_current_delta_empty_table_returns_zero(self):
        conn = init_database(":memory:")
        delta = read_current_delta(conn)
        assert delta == D("0.00000000")

    def test_write_and_read_delta(self):
        conn = init_database(":memory:")
        conn.execute("BEGIN")
        write_delta_in_txn(conn, D("0.45"), D("1000000"))
        conn.execute("COMMIT")
        delta = read_current_delta(conn)
        assert delta == D("0.45000000")

    def test_write_delta_formats_to_8_places(self):
        conn = init_database(":memory:")
        conn.execute("BEGIN")
        write_delta_in_txn(conn, D("0.333333333333"), D("500000"))
        conn.execute("COMMIT")
        delta = read_current_delta(conn)
        # Must be exactly 8 decimal places
        assert len(str(delta).split(".")[1]) == 8

    def test_most_recent_delta_returned(self):
        conn = init_database(":memory:")
        for d_val in ["0.10", "0.20", "0.45"]:
            conn.execute("BEGIN")
            write_delta_in_txn(conn, D(d_val), D("1000000"))
            conn.execute("COMMIT")
        delta = read_current_delta(conn)
        assert delta == D("0.45000000")


# ---------------------------------------------------------------------------
# Feed staleness
# ---------------------------------------------------------------------------

class TestFeedStaleness:
    """UAT-VAL-3.2: feed staleness detection."""

    def test_staleness_injected(self):
        result = asyncio.run(
            run_feed_with_staleness_detection(
                heartbeat_delay=0.01,
                inject_staleness_at=2,
                n_ticks=10,
            )
        )
        assert result["staleness_detected"] is True

    def test_pipeline_suspended_on_staleness(self):
        result = asyncio.run(
            run_feed_with_staleness_detection(
                heartbeat_delay=0.01,
                inject_staleness_at=2,
                n_ticks=10,
            )
        )
        assert result["pipeline_suspended"] is True

    def test_ticks_received_before_staleness(self):
        result = asyncio.run(
            run_feed_with_staleness_detection(
                heartbeat_delay=0.01,
                inject_staleness_at=3,
                n_ticks=10,
            )
        )
        # Should have received ticks before the stale event
        assert result["ticks_received"] >= 3

    def test_no_staleness_without_injection(self):
        result = asyncio.run(
            run_feed_with_staleness_detection(
                heartbeat_delay=0.01,
                inject_staleness_at=None,
                n_ticks=5,
            )
        )
        assert result["staleness_detected"] is False
        assert result["pipeline_suspended"] is False
        assert result["ticks_received"] == 5


# ---------------------------------------------------------------------------
# Broker transmission and atomic rollback
# ---------------------------------------------------------------------------

class TestAtomicRollback:
    """UAT-VAL-4.2: atomic rollback on broker failure."""

    def _make_conn_with_delta(self, initial_delta: str = "0.45") -> sqlite3.Connection:
        conn = init_database(":memory:")
        conn.execute("BEGIN")
        write_delta_in_txn(conn, D(initial_delta), D("1000000"), "initial")
        conn.execute("COMMIT")
        return conn

    def test_successful_commit_updates_delta(self):
        conn = self._make_conn_with_delta("0.30")
        result = asyncio.run(
            execute_rebalance_with_rollback(
                conn, D("0.55"), D("1000000"), inject_broker_failure=False
            )
        )
        assert result["committed"] is True
        assert read_current_delta(conn) == D("0.55000000")

    def test_broker_failure_triggers_rollback(self):
        conn = self._make_conn_with_delta("0.45")
        result = asyncio.run(
            execute_rebalance_with_rollback(
                conn, D("0.55"), D("1000000"), inject_broker_failure=True
            )
        )
        assert result["committed"] is False
        assert result["exception"] is not None

    def test_zero_delta_drift_after_rollback(self):
        conn = self._make_conn_with_delta("0.45")
        result = asyncio.run(
            execute_rebalance_with_rollback(
                conn, D("0.55"), D("1000000"), inject_broker_failure=True
            )
        )
        assert result["zero_drift"] is True

    def test_pre_attempt_delta_preserved(self):
        conn = self._make_conn_with_delta("0.45")
        result = asyncio.run(
            execute_rebalance_with_rollback(
                conn, D("0.55"), D("1000000"), inject_broker_failure=True
            )
        )
        assert result["pre_attempt_delta"] == D("0.45000000")
        assert result["post_attempt_delta"] == D("0.45000000")

    def test_rollback_result_keys_present(self):
        conn = self._make_conn_with_delta("0.45")
        result = asyncio.run(
            execute_rebalance_with_rollback(
                conn, D("0.55"), D("1000000"), inject_broker_failure=True
            )
        )
        for key in [
            "committed", "exception", "pre_attempt_delta",
            "post_attempt_delta", "delta_drift_after_rollback", "zero_drift"
        ]:
            assert key in result


# ---------------------------------------------------------------------------
# Full resilience experiment
# ---------------------------------------------------------------------------

class TestResilienceExperiment:
    """Integration test for the full resilience experiment."""

    def test_pass_rate_is_100(self):
        result = asyncio.run(run_resilience_experiment())
        assert result["pass_rate_pct"] == 100.0

    def test_uat_3_2_passes(self):
        result = asyncio.run(run_resilience_experiment())
        assert result["uat_val_3_2"]["pass"] is True

    def test_uat_4_2_passes(self):
        result = asyncio.run(run_resilience_experiment())
        assert result["uat_val_4_2"]["pass"] is True

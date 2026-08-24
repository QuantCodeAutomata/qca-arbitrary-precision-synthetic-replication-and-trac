"""Tests for src/monte_carlo.py — GBM simulation and buffered note payoff."""
import pytest
from decimal import Decimal

from src.monte_carlo import (
    buffered_note_payoff,
    gbm_path,
    run_monte_carlo_simulation,
)
from src.precision import D, ONE, ZERO, quantize8


class TestBufferedNotePayoff:
    """Tests for the downside-buffered note payoff function."""

    S0 = D("400")

    def test_upside_full_participation(self):
        """If S_T > S_0, investor gets full upside."""
        S_T = D("440")  # +10% gain
        payoff = buffered_note_payoff(S_T, self.S0)
        # ratio = 440/400 = 1.10 → payoff = 400 * 1.10 = 440
        assert payoff == quantize8(S_T)

    def test_buffer_zone_no_loss(self):
        """If S_T within buffer (< 10% loss), investor gets full notional back."""
        S_T = D("365")  # -8.75% → within 10% buffer
        payoff = buffered_note_payoff(S_T, self.S0)
        assert payoff == quantize8(self.S0)  # protected

    def test_below_barrier_loss_passthrough(self):
        """Below 20% barrier, losses beyond the buffer pass through."""
        S_T = D("300")  # -25% → below barrier, beyond 10% buffer
        payoff = buffered_note_payoff(S_T, self.S0)
        # return_pct = -0.25, payoff_ratio = 1 + (-0.25) + 0.10 = 0.85
        expected = quantize8(D("0.85") * self.S0)
        assert payoff == expected

    def test_at_barrier_threshold(self):
        """Exactly at barrier level → small loss (depends on buffer config)."""
        S_T = D("320")  # exactly -20% = barrier
        payoff = buffered_note_payoff(S_T, self.S0)
        # return_pct = -0.20 → below buffer (-0.10) → passthrough
        # payoff_ratio = 1 + (-0.20) + 0.10 = 0.90
        expected = quantize8(D("0.90") * self.S0)
        assert payoff == expected

    def test_at_zero_no_negative_payoff(self):
        """Even at S_T = 0 the payoff is 100% - (1 - buffer). Not less than zero
        unless barrier < buffer."""
        S_T = D("0.01")
        payoff = buffered_note_payoff(S_T, self.S0)
        # return_pct ≈ -1 → payoff_ratio = 1 + (-1) + 0.10 = 0.10
        assert payoff >= D("0")

    def test_payoff_is_decimal(self):
        payoff = buffered_note_payoff(D("400"), self.S0)
        assert isinstance(payoff, Decimal)

    def test_payoff_has_8_decimal_places(self):
        payoff = buffered_note_payoff(D("450"), self.S0)
        # Should be quantized to 8 d.p.
        assert payoff == quantize8(payoff)

    def test_exact_notional_returned_flat(self):
        """If S_T == S_0 exactly, return equals S_0 (0% return → in buffer zone)."""
        payoff = buffered_note_payoff(self.S0, self.S0)
        assert payoff == quantize8(self.S0)


class TestGBMPath:
    """Tests for GBM price path generator."""

    def test_path_length(self):
        path = gbm_path(D("100"), D("0.05"), D("0.20"), D("1"), steps=252, seed=1)
        assert len(path) == 253  # initial + 252 steps

    def test_path_starts_at_s0(self):
        S0 = D("200")
        path = gbm_path(S0, D("0.05"), D("0.20"), D("1"), steps=10, seed=99)
        assert path[0] == S0

    def test_path_is_positive(self):
        path = gbm_path(D("100"), D("0.05"), D("0.20"), D("1"), steps=252, seed=7)
        assert all(p > ZERO for p in path)

    def test_path_is_decimal_list(self):
        path = gbm_path(D("100"), D("0.05"), D("0.20"), D("1"), steps=5, seed=0)
        assert all(isinstance(p, Decimal) for p in path)

    def test_reproducibility_with_seed(self):
        path1 = gbm_path(D("100"), D("0.05"), D("0.20"), D("1"), steps=20, seed=42)
        path2 = gbm_path(D("100"), D("0.05"), D("0.20"), D("1"), steps=20, seed=42)
        assert path1 == path2

    def test_different_seeds_differ(self):
        path1 = gbm_path(D("100"), D("0.05"), D("0.20"), D("1"), steps=20, seed=1)
        path2 = gbm_path(D("100"), D("0.05"), D("0.20"), D("1"), steps=20, seed=2)
        assert path1 != path2


class TestMonteCarloSimulation:
    """Integration tests for the Monte Carlo simulation runner."""

    _small_results = None  # cache to avoid re-running

    @classmethod
    def get_results(cls):
        if cls._small_results is None:
            cls._small_results = run_monte_carlo_simulation(
                S0=D("400"),
                r=D("0.05"),
                sigma=D("0.18"),
                T=D("1"),
                steps=50,
                n_paths=200,
                seed=7,
            )
        return cls._small_results

    def test_returns_required_keys(self):
        res = self.get_results()
        for key in [
            "n_paths", "tracking_accuracy_rate", "mean_tracking_error",
            "max_tracking_error", "mean_rebalances_per_path",
            "theoretical_payoffs", "synthetic_payoffs",
        ]:
            assert key in res, f"Missing key: {key}"

    def test_n_paths_matches_request(self):
        res = self.get_results()
        assert res["n_paths"] == 200

    def test_tracking_accuracy_rate_between_0_and_100(self):
        res = self.get_results()
        tar = float(res["tracking_accuracy_rate"])
        assert 0.0 <= tar <= 100.0, f"Tracking accuracy rate {tar} out of [0, 100]"

    def test_mean_tracking_error_is_decimal(self):
        res = self.get_results()
        assert isinstance(res["mean_tracking_error"], Decimal)

    def test_max_tracking_error_ge_mean(self):
        res = self.get_results()
        assert res["max_tracking_error"] >= res["mean_tracking_error"]

    def test_mean_rebalances_positive(self):
        res = self.get_results()
        assert res["mean_rebalances_per_path"] >= D("0")

    def test_payoff_lists_nonempty(self):
        res = self.get_results()
        assert len(res["theoretical_payoffs"]) > 0
        assert len(res["synthetic_payoffs"]) > 0

    def test_payoffs_are_positive(self):
        res = self.get_results()
        for p in res["theoretical_payoffs"]:
            assert p >= D("0"), f"Negative theoretical payoff: {p}"

    def test_reproducibility(self):
        r1 = run_monte_carlo_simulation(D("400"), D("0.05"), D("0.18"), D("1"), 10, 50, seed=99)
        r2 = run_monte_carlo_simulation(D("400"), D("0.05"), D("0.18"), D("1"), 10, 50, seed=99)
        assert r1["mean_tracking_error"] == r2["mean_tracking_error"]

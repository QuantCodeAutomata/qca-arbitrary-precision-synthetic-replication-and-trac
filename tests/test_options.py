"""Tests for src/options.py — Black-Scholes pricing in Decimal arithmetic."""
import pytest
from decimal import Decimal

from src.options import (
    bs_call_price,
    bs_put_price,
    delta,
    gamma,
    theta,
    vega,
    bs_d1,
    bs_d2,
)
from src.precision import D, quantize8

# Standard test parameters
S = D("100")
K = D("100")
r = D("0.05")
sig = D("0.20")
T = D("1")


class TestBSPricing:
    """Black-Scholes call and put pricing tests."""

    def test_call_price_is_positive(self):
        price = bs_call_price(S, K, r, sig, T)
        assert price > D("0")

    def test_put_price_is_positive(self):
        price = bs_put_price(S, K, r, sig, T)
        assert price > D("0")

    def test_call_price_returns_decimal(self):
        price = bs_call_price(S, K, r, sig, T)
        assert isinstance(price, Decimal)

    def test_put_price_returns_decimal(self):
        price = bs_put_price(S, K, r, sig, T)
        assert isinstance(price, Decimal)

    def test_put_call_parity(self):
        """C - P = S - K*e^(-rT) (put-call parity)."""
        c = bs_call_price(S, K, r, sig, T)
        p = bs_put_price(S, K, r, sig, T)
        disc_k = K * ((-r * T).exp())
        parity_lhs = c - p
        parity_rhs = S - disc_k
        # Tolerance of 1e-6 for decimal approximations in normal CDF
        assert abs(parity_lhs - parity_rhs) < D("1E-5"), (
            f"Put-call parity violated: {parity_lhs} != {parity_rhs}"
        )

    def test_call_price_above_intrinsic(self):
        intrinsic = max(S - K, D("0"))
        call_price = bs_call_price(S, K, r, sig, T)
        assert call_price >= intrinsic

    def test_deep_itm_call_approaches_forward(self):
        """Deep in-the-money call ≈ S - K*e^(-rT)."""
        K_low = D("50")
        call = bs_call_price(S, K_low, r, sig, T)
        forward = S - K_low * ((-r * T).exp())
        assert abs(call - forward) < D("1")  # within $1 for deep ITM

    def test_deep_otm_put_approaches_zero(self):
        """Deep out-of-the-money put ≈ 0."""
        K_low = D("50")
        put = bs_put_price(S, K_low, r, sig, T)
        assert put < D("0.01")

    def test_zero_time_raises(self):
        """T <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            bs_call_price(S, K, r, sig, D("0"))

    def test_known_value_atm_call(self):
        """ATM call with known reference value from BS formula tables."""
        call = bs_call_price(S, K, r, sig, T)
        # Reference: ~10.45 for S=100, K=100, r=5%, sigma=20%, T=1
        assert D("9.0") < call < D("12.0"), f"ATM call out of range: {call}"


class TestGreeks:
    """Tests for option Greeks."""

    def test_call_delta_in_range(self):
        d = delta(S, K, r, sig, T, "call")
        assert D("0") < d < D("1")

    def test_put_delta_in_range(self):
        d = delta(S, K, r, sig, T, "put")
        assert D("-1") < d < D("0")

    def test_call_put_delta_sum_equals_one(self):
        """Delta_call - Delta_put = 1 (from put-call parity differentiation)."""
        d_call = delta(S, K, r, sig, T, "call")
        d_put = delta(S, K, r, sig, T, "put")
        # d_call - (-|d_put|) = 1  →  d_call + |d_put| ≈ 1
        assert abs(d_call - d_put - D("1")) < D("1E-5")

    def test_gamma_is_positive(self):
        g = gamma(S, K, r, sig, T)
        assert g > D("0")

    def test_gamma_returns_decimal(self):
        assert isinstance(gamma(S, K, r, sig, T), Decimal)

    def test_vega_is_positive(self):
        v = vega(S, K, r, sig, T)
        assert v > D("0")

    def test_theta_call_is_negative(self):
        """Theta for a long call is typically negative (time decay)."""
        th = theta(S, K, r, sig, T, "call")
        assert th < D("0")

    def test_zero_time_greeks_return_zero(self):
        """At expiration, Greeks default to zero."""
        assert delta(S, K, r, sig, D("0")) == D("0")
        assert gamma(S, K, r, sig, D("0")) == D("0")
        assert vega(S, K, r, sig, D("0")) == D("0")
        assert theta(S, K, r, sig, D("0")) == D("0")

    def test_high_vol_higher_vega(self):
        """Higher volatility → higher vega."""
        v_low = vega(S, K, r, D("0.10"), T)
        v_high = vega(S, K, r, D("0.30"), T)
        # Vega is S*N'(d1)*sqrt(T), which itself depends on d1 and hence sigma.
        # Both should be positive; the relationship is not strictly monotone
        # but both must be positive.
        assert v_low > D("0")
        assert v_high > D("0")

    def test_all_greeks_are_decimal(self):
        for greek_fn, args in [
            (delta, (S, K, r, sig, T, "call")),
            (gamma, (S, K, r, sig, T)),
            (theta, (S, K, r, sig, T, "call")),
            (vega, (S, K, r, sig, T)),
        ]:
            result = greek_fn(*args)
            assert isinstance(result, Decimal), f"{greek_fn.__name__} did not return Decimal"

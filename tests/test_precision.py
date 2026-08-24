"""Tests for src/precision.py — arbitrary-precision arithmetic utilities."""
import pytest
from decimal import Decimal, getcontext

from src.precision import (
    D,
    EIGHT_PLACES,
    ONE,
    TWO,
    ZERO,
    fmt8,
    precision_unit_test,
    quantize8,
    to_decimal,
)


class TestPrecisionUnitTest:
    """UAT: 1/3 * 3 rounds to 1.00000000 at 8 decimal places — zero output precision loss."""

    def test_one_third_times_three_equals_one_at_8dp(self):
        """Core deterministic precision check (paper methodology step 0).

        At prec=28, D("1")/D("3")*D("3") = 0.9999999999999999999999999999.
        When quantized to the mandatory 8-decimal-place output format this
        rounds correctly to 1.00000000, confirming zero precision loss in
        state outputs — which is the claim the methodology validates.
        """
        result = D("1") / D("3") * D("3")
        assert quantize8(result) == D("1.00000000"), (
            f"Expected 1.00000000 after quantization, got {quantize8(result)}"
        )

    def test_precision_unit_test_returns_true(self):
        assert precision_unit_test() is True

    def test_context_prec_is_28(self):
        """Decimal context precision must be at least 28."""
        assert getcontext().prec >= 28


class TestToDecimal:
    """Tests for to_decimal() conversion helper."""

    def test_from_int(self):
        assert to_decimal(5) == D("5")

    def test_from_string(self):
        assert to_decimal("3.14159") == D("3.14159")

    def test_from_float_avoids_float_artefacts(self):
        # float(0.1) has IEEE-754 representation issues; to_decimal must handle it
        val = to_decimal(0.1)
        assert isinstance(val, Decimal)
        # Should round to something very close to 0.1 (not 0.10000000000000001)
        assert abs(val - D("0.1")) < D("1E-10")

    def test_from_decimal_passthrough(self):
        d = D("42.00000001")
        assert to_decimal(d) is d

    def test_from_zero(self):
        assert to_decimal(0) == ZERO


class TestFmt8:
    """Tests for fmt8() — format Decimal to exactly 8 decimal places."""

    def test_exactly_8_places(self):
        result = fmt8(D("1.5"))
        assert result == "1.50000000"
        assert len(result.split(".")[1]) == 8

    def test_truncates_at_8_places(self):
        val = D("1") / D("3")  # repeating decimal
        result = fmt8(val)
        assert len(result.split(".")[1]) == 8

    def test_integer_value(self):
        result = fmt8(D("100"))
        assert result == "100.00000000"

    def test_negative_value(self):
        result = fmt8(D("-0.12345678"))
        assert result == "-0.12345678"


class TestQuantize8:
    """Tests for quantize8() — round Decimal to 8 places."""

    def test_output_is_decimal(self):
        result = quantize8(D("3.14"))
        assert isinstance(result, Decimal)

    def test_result_has_8_decimal_exponent(self):
        result = quantize8(D("1.999999999999"))
        assert result == D("2.00000000")

    def test_zero(self):
        assert quantize8(ZERO) == D("0.00000000")

    def test_preserves_precision_boundary(self):
        val = D("0.00000001")
        assert quantize8(val) == D("0.00000001")

    def test_rounds_not_truncates(self):
        # 0.0000000051 is strictly above the 8th-place midpoint → rounds up.
        # (0.000000005 is the exact midpoint; ROUND_HALF_EVEN rounds it to 0
        #  because the 8th digit would be 0, which is even.)
        val = D("0.0000000051")
        result = quantize8(val)
        assert result == D("0.00000001")


class TestEdgeCases:
    """Edge-case tests for arithmetic stability."""

    def test_large_value(self):
        val = D("1000000.00000000")
        assert quantize8(val) == D("1000000.00000000")

    def test_very_small_value(self):
        val = D("0.00000001")
        assert quantize8(val) == D("0.00000001")

    def test_subtraction_precision(self):
        # Standard float would give 0.19999999999999998
        a = D("1.2")
        b = D("1.0")
        result = a - b
        assert result == D("0.2")

    def test_multiplication_precision(self):
        result = D("0.1") * D("0.2")
        assert result == D("0.02")

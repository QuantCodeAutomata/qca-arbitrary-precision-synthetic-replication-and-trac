"""Tests for src/savings_ledger.py — cost drag analysis, ledger, paired t-test."""
import pytest
from decimal import Decimal

import pandas as pd

from src.savings_ledger import (
    build_savings_ledger,
    compute_bank_note_cost,
    paired_ttest_yields,
    run_savings_experiment,
    DEFAULT_BANK_FEE_PCT,
    DEFAULT_BID_ASK_SPREAD,
    DEFAULT_DIST_FEE,
)
from src.precision import D, quantize8


class TestBankNoteCost:
    """Tests for compute_bank_note_cost()."""

    def test_positive_cost(self):
        cost = compute_bank_note_cost(D("1000000"), D("5"))
        assert cost > D("0")

    def test_cost_is_decimal(self):
        cost = compute_bank_note_cost(D("1000000"), D("5"))
        assert isinstance(cost, Decimal)

    def test_cost_quantized_to_8_places(self):
        cost = compute_bank_note_cost(D("1000000"), D("5"))
        assert cost == quantize8(cost)

    def test_cost_includes_all_components(self):
        notional = D("1000000")
        T = D("5")
        cost = compute_bank_note_cost(notional, T)
        # Annual fee: 1000000 * 0.0150 * 5 = 75000
        # Bid-ask:    1000000 * 0.0030      = 3000
        # Dist fee:   1000000 * 0.0100      = 10000
        # Total: 88000
        expected = quantize8(D("88000"))
        assert cost == expected

    def test_larger_notional_higher_cost(self):
        c1 = compute_bank_note_cost(D("1000000"), D("5"))
        c2 = compute_bank_note_cost(D("2000000"), D("5"))
        assert c2 == c1 * 2

    def test_longer_tenor_higher_cost(self):
        c1 = compute_bank_note_cost(D("1000000"), D("1"))
        c5 = compute_bank_note_cost(D("1000000"), D("5"))
        assert c5 > c1


class TestBuildSavingsLedger:
    """Tests for build_savings_ledger()."""

    def _make_inputs(self, n: int = 5):
        gross = [D("1100000")] * n
        syn_cost = [D("500")] * n
        bank_cost = [D("88000")] * n
        return gross, syn_cost, bank_cost

    def test_returns_dataframe(self):
        gross, syn, bank = self._make_inputs()
        df = build_savings_ledger(syn, gross, bank)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        gross, syn, bank = self._make_inputs()
        df = build_savings_ledger(syn, gross, bank)
        for col in [
            "gross_payoff", "synthetic_cost", "synthetic_net_yield",
            "bank_cost", "bank_net_yield", "execution_spread",
            "savings", "cumulative_savings",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_matches_input(self):
        n = 7
        gross, syn, bank = self._make_inputs(n)
        df = build_savings_ledger(syn, gross, bank)
        assert len(df) == n

    def test_execution_spread_is_positive(self):
        """Bank cost > synthetic cost → execution spread > 0."""
        gross, syn, bank = self._make_inputs()
        df = build_savings_ledger(syn, gross, bank)
        assert (df["execution_spread"] > 0).all()

    def test_cumulative_savings_monotonically_increasing(self):
        gross, syn, bank = self._make_inputs()
        df = build_savings_ledger(syn, gross, bank)
        assert (df["cumulative_savings"].diff().dropna() >= 0).all()

    def test_synthetic_yield_equals_gross_minus_syn_cost(self):
        gross = [D("1100000")]
        syn = [D("500")]
        bank = [D("88000")]
        df = build_savings_ledger(syn, gross, bank)
        expected_yield = float(quantize8(D("1100000") - D("500")))
        assert abs(df["synthetic_net_yield"].iloc[0] - expected_yield) < 1e-5


class TestPairedTtest:
    """Tests for paired_ttest_yields()."""

    def test_significant_difference(self):
        """Synthetic yields consistently higher → should be significant."""
        syn = [1_100_000 + i * 100 for i in range(30)]
        bank = [1_000_000 + i * 100 for i in range(30)]
        result = paired_ttest_yields(syn, bank)
        assert result["significant_at_5pct"] is True

    def test_no_difference(self):
        """Identical yields → p-value = 1, not significant."""
        yields = [1_000_000.0] * 30
        result = paired_ttest_yields(yields, yields[:])
        assert result["p_value"] > 0.05
        assert result["significant_at_5pct"] is False

    def test_returns_required_keys(self):
        result = paired_ttest_yields([1.1, 1.2, 1.3], [1.0, 1.0, 1.0])
        for key in ["t_statistic", "p_value", "mean_alpha", "significant_at_5pct"]:
            assert key in result

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            paired_ttest_yields([1.0, 2.0], [1.0])

    def test_single_element_raises(self):
        with pytest.raises(ValueError):
            paired_ttest_yields([1.0], [1.0])

    def test_mean_alpha_correct_sign(self):
        """Synthetic outperforms bank → positive mean alpha."""
        result = paired_ttest_yields([1.1, 1.2], [1.0, 1.0])
        assert result["mean_alpha"] > 0

    def test_t_statistic_is_float(self):
        result = paired_ttest_yields([1.1, 1.2, 1.3], [1.0, 1.0, 1.0])
        assert isinstance(result["t_statistic"], float)


class TestSavingsExperiment:
    """Integration tests for run_savings_experiment()."""

    _result = None

    @classmethod
    def get_result(cls):
        if cls._result is None:
            cls._result = run_savings_experiment(
                n_paths=100,
                notional=D("1000000"),
                T_years=D("5"),
                avg_rebalance_cost_bps=D("0.0001"),
                n_rolling_regimes=3,
                seed=42,
            )
        return cls._result

    def test_returns_required_keys(self):
        res = self.get_result()
        for key in ["regime_results", "summary_df", "all_regimes_significant",
                    "grand_mean_alpha", "grand_mean_savings"]:
            assert key in res

    def test_regime_count(self):
        res = self.get_result()
        assert len(res["regime_results"]) == 3

    def test_all_regimes_significant(self):
        """With clear synthetic cost advantage, all regimes should be significant."""
        res = self.get_result()
        assert res["all_regimes_significant"] is True

    def test_grand_mean_alpha_positive(self):
        """Synthetic outperforms bank → positive grand mean alpha."""
        res = self.get_result()
        assert res["grand_mean_alpha"] > 0

    def test_grand_mean_savings_positive(self):
        res = self.get_result()
        assert res["grand_mean_savings"] > 0

    def test_summary_df_is_dataframe(self):
        res = self.get_result()
        assert isinstance(res["summary_df"], pd.DataFrame)

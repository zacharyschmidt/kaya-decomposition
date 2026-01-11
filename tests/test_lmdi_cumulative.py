"""Tests for cumulative LMDI functions."""

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from kaya_decomposition import (
    compute_kaya_variables,
    compute_kaya_factors,
    lmdi_cumulative as lmdi_names,
    kaya_variables as kaya_var_names,
)
from kaya_decomposition.lmdi_cumulative import (
    compute_lmdi_cumulative,
    compute_lmdi_cumulative_sum,
    _logarithmic_mean,
)


class TestLogarithmicMean:
    """Tests for the logarithmic mean helper function."""

    def test_logarithmic_mean_basic(self):
        """Test basic logarithmic mean calculation."""
        # L(2, 8) = (2 - 8) / (ln(2) - ln(8)) = -6 / (0.693 - 2.079) = 4.328
        result = _logarithmic_mean(2, 8)
        assert np.isclose(result, 4.328, rtol=0.01)

    def test_logarithmic_mean_equal_values(self):
        """Test that L(a, a) = a."""
        result = _logarithmic_mean(5, 5)
        assert np.isclose(result, 5)

    def test_logarithmic_mean_array(self):
        """Test logarithmic mean with array inputs."""
        a = np.array([2, 5, 10])
        b = np.array([8, 5, 20])
        result = _logarithmic_mean(a, b)
        assert len(result) == 3
        assert np.isclose(result[1], 5)  # equal values

    def test_logarithmic_mean_nonpositive(self):
        """Test that non-positive values return NaN."""
        assert np.isnan(_logarithmic_mean(0, 5))
        assert np.isnan(_logarithmic_mean(-1, 5))
        assert np.isnan(_logarithmic_mean(5, 0))

    def test_logarithmic_mean_symmetric(self):
        """Test that logarithmic mean is symmetric: L(a,b) = L(b,a)."""
        result1 = _logarithmic_mean(3, 7)
        result2 = _logarithmic_mean(7, 3)
        assert np.isclose(result1, result2)


class TestComputeLmdiCumulative:
    """Tests for compute_lmdi_cumulative function."""

    def test_returns_iamdataframe(self, multi_year_dataframe):
        """Test that function returns an IamDataFrame."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        result = compute_lmdi_cumulative(factors, base_year=2020)
        assert isinstance(result, IamDataFrame)

    def test_base_year_has_zero_contribution(self, multi_year_dataframe):
        """Test that base year has zero LMDI contributions."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        result = compute_lmdi_cumulative(factors, base_year=2020)

        # All factors should be zero at base year
        base_year_data = result.filter(year=2020).data
        assert np.allclose(base_year_data["value"], 0)

    def test_contributions_sum_to_tfc_diff(self, multi_year_dataframe):
        """Test that LMDI contributions sum to TFC difference from base year."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        result = compute_lmdi_cumulative(factors, base_year=2020)

        # Get TFC values
        tfc = factors.filter(variable=kaya_var_names.TFC)
        tfc_base = tfc.filter(year=2020).data["value"].values[0]

        for year in [2030, 2040, 2050]:
            tfc_year = tfc.filter(year=year).data["value"].values[0]
            tfc_diff = tfc_year - tfc_base

            # Sum all LMDI contributions for this year
            year_data = result.filter(year=year).data
            contribution_sum = year_data["value"].sum()

            assert np.isclose(contribution_sum, tfc_diff, rtol=0.01), \
                f"Year {year}: sum={contribution_sum}, tfc_diff={tfc_diff}"

    def test_has_expected_variables(self, multi_year_dataframe):
        """Test that output contains all expected LMDI variables."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        result = compute_lmdi_cumulative(factors, base_year=2020)

        expected_vars = [
            lmdi_names.Pop_cumulative,
            lmdi_names.GNP_per_P_cumulative,
            lmdi_names.FE_per_GNP_cumulative,
            lmdi_names.PEdeq_per_FE_cumulative,
            lmdi_names.PEFF_per_PEDEq_cumulative,
            lmdi_names.TFC_per_PEFF_cumulative,
        ]

        for var in expected_vars:
            assert var in result.variable, f"Missing variable: {var}"

    def test_all_contributions_non_negative_after_base_year(self, multi_year_dataframe):
        """Test that LMDI contributions are non-negative for years after base year.

        The non-negativity correction only applies to years where TFC increases
        from the base year. Years before the base year naturally have negative
        contributions since TFC was lower in the past.
        """
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        result = compute_lmdi_cumulative(factors, base_year=2020)

        # Only check years >= base_year for non-negativity
        post_base_data = result.filter(year=[2020, 2030, 2040, 2050]).data
        assert (post_base_data["value"] >= -1e-10).all(), \
            "Found negative values in corrected LMDI for years after base year"

    def test_scenario_filter(self, multi_year_dataframe):
        """Test filtering to specific scenario."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        result = compute_lmdi_cumulative(
            factors,
            base_year=2020,
            scenario=("TestModel", "TestScenario", "World")
        )
        assert isinstance(result, IamDataFrame)
        assert len(result.data) > 0


class TestComputeLmdiCumulativeSum:
    """Tests for compute_lmdi_cumulative_sum function."""

    def test_returns_dataframe(self, multi_year_dataframe):
        """Test that function returns a pandas DataFrame."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        lmdi = compute_lmdi_cumulative(factors, base_year=2020)
        result = compute_lmdi_cumulative_sum(lmdi)
        assert isinstance(result, pd.DataFrame)

    def test_default_periods(self, multi_year_dataframe):
        """Test that default periods are used correctly."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        lmdi = compute_lmdi_cumulative(factors, base_year=2020)
        result = compute_lmdi_cumulative_sum(lmdi)

        # Should have columns for default periods
        assert "2020 to 2050" in result.columns

    def test_custom_periods(self, multi_year_dataframe):
        """Test with custom period specification."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        lmdi = compute_lmdi_cumulative(factors, base_year=2020)

        result = compute_lmdi_cumulative_sum(
            lmdi,
            periods=[(2020, 2030), (2030, 2050)]
        )

        assert "2020 to 2030" in result.columns
        assert "2030 to 2050" in result.columns

    def test_row_labels(self, multi_year_dataframe):
        """Test that output has correct row labels."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        lmdi = compute_lmdi_cumulative(factors, base_year=2020)
        result = compute_lmdi_cumulative_sum(lmdi)

        expected_rows = [
            lmdi_names.Pop_cumulative,
            lmdi_names.GNP_per_P_cumulative,
            lmdi_names.FE_per_GNP_cumulative,
        ]

        for row in expected_rows:
            assert row in result.index

    def test_period_sums_are_cumulative(self, multi_year_dataframe):
        """Test that 2020-2050 sum equals sum of individual year contributions."""
        kaya_vars = compute_kaya_variables(multi_year_dataframe)
        factors = compute_kaya_factors(kaya_vars)
        lmdi = compute_lmdi_cumulative(factors, base_year=2020)

        result = compute_lmdi_cumulative_sum(
            lmdi,
            periods=[(2020, 2050)]
        )

        # Sum individual contributions for 2030, 2040, 2050 (excluding 2020 base year)
        for var in result.index:
            var_data = lmdi.filter(variable=var).data
            manual_sum = var_data[var_data["year"].isin([2030, 2040, 2050])]["value"].sum()
            assert np.isclose(result.loc[var, "2020 to 2050"], manual_sum, rtol=0.01)

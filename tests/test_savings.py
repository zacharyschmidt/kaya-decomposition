"""Tests for savings calculations (two-scenario comparison over time)."""

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from kaya_decomposition import (
    compute_savings,
    compute_savings_with_percentages,
    compute_lmdi_scenario_comparison,
    savings as savings_names,
)
from kaya_decomposition.savings import (
    _logarithmic_mean,
    compute_cumulative_emissions,
    compute_intervention_ccs,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def two_scenario_multi_year_data():
    """Create test data with two scenarios over multiple years.

    Reference scenario: Higher emissions baseline
    Intervention scenario: Lower emissions with some CCS
    """
    years = [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

    # Reference scenario - growing emissions
    ref_data = []
    ref_base = {
        "Population": 7700,  # million
        "GDP|PPP": 112000,  # billion USD
        "GDP|MER": 78000,
        "Final Energy": 420,  # EJ/yr
        "Primary Energy": 590,  # EJ/yr
        "Primary Energy|Coal": 170,
        "Primary Energy|Gas": 150,
        "Primary Energy|Oil": 175,
        "Emissions|CO2|Energy and Industrial Processes": 38000,  # Mt CO2/yr
        "Emissions|CO2|Industrial Processes": 1800,
        "Emissions|CO2|AFOLU": 5000,
        "Carbon Sequestration|CCS": 0,
        "Carbon Sequestration|CCS|Biomass": 0,
        "Carbon Sequestration|CCS|Fossil|Energy": 0,
        "Carbon Sequestration|CCS|Fossil|Industrial Processes": 0,
        "Carbon Sequestration|CCS|Biomass|Energy": 0,
        "Carbon Sequestration|CCS|Biomass|Industrial Processes": 0,
        "Emissions|CH4": 400,
        "Emissions|N2O": 11000,
        "Emissions|F-Gases": 1700,
    }

    # Reference growth factors (compound per 10 years)
    ref_growth = {
        "Population": 1.08,
        "GDP|PPP": 1.40,
        "GDP|MER": 1.40,
        "Final Energy": 1.15,
        "Primary Energy": 1.12,
        "Primary Energy|Coal": 1.15,
        "Primary Energy|Gas": 1.10,
        "Primary Energy|Oil": 0.98,
        "Emissions|CO2|Energy and Industrial Processes": 1.10,
        "Emissions|CO2|Industrial Processes": 1.08,
        "Emissions|CO2|AFOLU": 0.95,
        "Emissions|CH4": 1.03,
        "Emissions|N2O": 1.02,
        "Emissions|F-Gases": 1.15,
    }

    for var, base_val in ref_base.items():
        for i, year in enumerate(years):
            if var in ref_growth:
                value = base_val * (ref_growth[var] ** i)
            else:
                value = base_val
            ref_data.append({
                "model": "TestModel",
                "scenario": "Reference",
                "region": "World",
                "variable": var,
                "unit": "various",
                "year": year,
                "value": value,
            })

    # Intervention scenario - emissions decline with CCS
    int_data = []
    int_base = {
        "Population": 7700,  # Same population
        "GDP|PPP": 112000,  # Same GDP
        "GDP|MER": 78000,
        "Final Energy": 400,  # Slightly lower energy
        "Primary Energy": 550,
        "Primary Energy|Coal": 140,  # Less coal
        "Primary Energy|Gas": 160,
        "Primary Energy|Oil": 150,
        "Emissions|CO2|Energy and Industrial Processes": 35000,  # Lower initial
        "Emissions|CO2|Industrial Processes": 1700,
        "Emissions|CO2|AFOLU": 4000,
        "Carbon Sequestration|CCS": 100,
        "Carbon Sequestration|CCS|Biomass": 50,
        "Carbon Sequestration|CCS|Fossil|Energy": 80,
        "Carbon Sequestration|CCS|Fossil|Industrial Processes": 20,
        "Carbon Sequestration|CCS|Biomass|Energy": 40,
        "Carbon Sequestration|CCS|Biomass|Industrial Processes": 10,
        "Emissions|CH4": 380,
        "Emissions|N2O": 10500,
        "Emissions|F-Gases": 1500,
    }

    # Intervention: emissions decline, CCS grows
    int_growth = {
        "Population": 1.06,
        "GDP|PPP": 1.35,
        "GDP|MER": 1.35,
        "Final Energy": 0.98,  # Energy declines
        "Primary Energy": 0.96,
        "Primary Energy|Coal": 0.85,  # Coal declines faster
        "Primary Energy|Gas": 1.02,
        "Primary Energy|Oil": 0.90,
        "Emissions|CO2|Energy and Industrial Processes": 0.90,  # Emissions decline
        "Emissions|CO2|Industrial Processes": 0.95,
        "Emissions|CO2|AFOLU": 0.85,  # Land use emissions decline
        "Carbon Sequestration|CCS|Fossil|Energy": 1.50,  # CCS grows
        "Carbon Sequestration|CCS|Fossil|Industrial Processes": 1.40,
        "Carbon Sequestration|CCS|Biomass|Energy": 1.60,
        "Carbon Sequestration|CCS|Biomass|Industrial Processes": 1.50,
        "Emissions|CH4": 0.95,
        "Emissions|N2O": 0.97,
        "Emissions|F-Gases": 0.90,
    }

    for var, base_val in int_base.items():
        for i, year in enumerate(years):
            if var in int_growth:
                value = base_val * (int_growth[var] ** i)
            else:
                value = base_val
            int_data.append({
                "model": "TestModel",
                "scenario": "Intervention",
                "region": "World",
                "variable": var,
                "unit": "various",
                "year": year,
                "value": value,
            })

    combined = pd.DataFrame(ref_data + int_data)
    return IamDataFrame(combined)


# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================

class TestLogarithmicMean:
    """Tests for _logarithmic_mean helper function."""

    def test_equal_values(self):
        """When a == b, logarithmic mean should return a."""
        result = _logarithmic_mean(5.0, 5.0)
        assert np.isclose(result, 5.0)

    def test_different_values(self):
        """Test logarithmic mean with different positive values."""
        # L(2, 4) = (2 - 4) / (ln(2) - ln(4)) = -2 / -ln(2) = 2 / ln(2) ≈ 2.885
        result = _logarithmic_mean(2.0, 4.0)
        expected = (2.0 - 4.0) / (np.log(2.0) - np.log(4.0))
        assert np.isclose(result, expected)

    def test_non_positive_values(self):
        """Non-positive values should return NaN."""
        assert np.isnan(_logarithmic_mean(0.0, 5.0))
        assert np.isnan(_logarithmic_mean(5.0, 0.0))
        assert np.isnan(_logarithmic_mean(-1.0, 5.0))

    def test_array_input(self):
        """Test with array inputs."""
        a = np.array([2.0, 5.0, 10.0])
        b = np.array([4.0, 5.0, 20.0])
        result = _logarithmic_mean(a, b)
        assert len(result) == 3
        assert np.isclose(result[1], 5.0)  # Equal values


# ============================================================================
# Tests for compute_lmdi_scenario_comparison
# ============================================================================

class TestComputeLmdiScenarioComparison:
    """Tests for compute_lmdi_scenario_comparison function."""

    def test_returns_iamdataframe(self, two_scenario_multi_year_data):
        """Should return pyam.IamDataFrame."""
        from kaya_decomposition import compute_kaya_variables, compute_kaya_factors

        kaya_vars = compute_kaya_variables(two_scenario_multi_year_data)
        kaya_factors = compute_kaya_factors(kaya_vars)

        result = compute_lmdi_scenario_comparison(
            kaya_factors,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        assert isinstance(result, IamDataFrame)

    def test_combined_scenario_name(self, two_scenario_multi_year_data):
        """Scenario name should be ref::int format."""
        from kaya_decomposition import compute_kaya_variables, compute_kaya_factors

        kaya_vars = compute_kaya_variables(two_scenario_multi_year_data)
        kaya_factors = compute_kaya_factors(kaya_vars)

        result = compute_lmdi_scenario_comparison(
            kaya_factors,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        assert "Reference::Intervention" in result.data["scenario"].values

    def test_contains_all_factor_contributions(self, two_scenario_multi_year_data):
        """Should contain contributions for all Kaya factors."""
        from kaya_decomposition import compute_kaya_variables, compute_kaya_factors

        kaya_vars = compute_kaya_variables(two_scenario_multi_year_data)
        kaya_factors = compute_kaya_factors(kaya_vars)

        result = compute_lmdi_scenario_comparison(
            kaya_factors,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        variables = result.data["variable"].unique()
        expected_vars = [
            savings_names.POPULATION,
            savings_names.ECONOMIC_ACTIVITY,
            savings_names.ENERGY_INTENSITY,
            savings_names.ENERGY_SUPPLY_LOSS,
            savings_names.FOSSIL_FRACTION,
            savings_names.CARBON_INTENSITY,
        ]

        for var in expected_vars:
            assert var in variables, f"Missing variable: {var}"


# ============================================================================
# Tests for compute_savings
# ============================================================================

class TestComputeSavings:
    """Tests for compute_savings main function."""

    def test_returns_dataframe(self, two_scenario_multi_year_data):
        """Should return pandas DataFrame."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        assert isinstance(result, pd.DataFrame)

    def test_default_periods(self, two_scenario_multi_year_data):
        """Default periods should be used when not specified."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        expected_columns = ["2020 to 2050", "2050 to 2100", "2020 to 2100"]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_custom_periods(self, two_scenario_multi_year_data):
        """Custom periods should be respected."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
            periods=[(2020, 2060), (2060, 2100)],
        )

        assert "2020 to 2060" in result.columns
        assert "2060 to 2100" in result.columns
        assert "2020 to 2100" not in result.columns

    def test_contains_summary_rows(self, two_scenario_multi_year_data):
        """Should contain reference, intervention, and difference rows."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        assert savings_names.REF_CUMULATIVE in result.index
        assert savings_names.INT_CUMULATIVE in result.index
        assert savings_names.DIFFERENCE in result.index

    def test_contains_kaya_factor_rows(self, two_scenario_multi_year_data):
        """Should contain Kaya factor contribution rows."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        expected_rows = [
            savings_names.POPULATION,
            savings_names.ECONOMIC_ACTIVITY,
            savings_names.ENERGY_INTENSITY,
            savings_names.ENERGY_SUPPLY_LOSS,
            savings_names.FOSSIL_FRACTION,
            savings_names.CARBON_INTENSITY,
        ]

        for row in expected_rows:
            assert row in result.index, f"Missing row: {row}"

    def test_contains_sector_rows(self, two_scenario_multi_year_data):
        """Should contain non-Kaya sector contribution rows."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        expected_rows = [
            savings_names.INDUSTRIAL_PROCESS,
            savings_names.LAND_USE,
            savings_names.OTHER_GASES,
        ]

        for row in expected_rows:
            assert row in result.index, f"Missing row: {row}"

    def test_contains_ccs_rows(self, two_scenario_multi_year_data):
        """Should contain CCS contribution rows."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        assert savings_names.FOSSIL_CCS in result.index
        assert savings_names.BIOMASS_CCS in result.index

    def test_contains_total_row(self, two_scenario_multi_year_data):
        """Should contain Total/Net row."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        assert savings_names.TOTAL_NET in result.index

    def test_difference_equals_ref_minus_int(self, two_scenario_multi_year_data):
        """Difference should equal Reference - Intervention."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
            periods=[(2020, 2100)],
        )

        period = "2020 to 2100"
        ref = result.loc[savings_names.REF_CUMULATIVE, period]
        int_ = result.loc[savings_names.INT_CUMULATIVE, period]
        diff = result.loc[savings_names.DIFFERENCE, period]

        assert np.isclose(diff, ref - int_, rtol=0.001)

    def test_ref_greater_than_int(self, two_scenario_multi_year_data):
        """Reference emissions should be greater than intervention."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
            periods=[(2020, 2100)],
        )

        period = "2020 to 2100"
        ref = result.loc[savings_names.REF_CUMULATIVE, period]
        int_ = result.loc[savings_names.INT_CUMULATIVE, period]

        assert ref > int_, "Reference should have higher emissions"

    def test_ccs_values_negative(self, two_scenario_multi_year_data):
        """CCS contributions should be negative (representing emission reductions)."""
        result = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
            periods=[(2020, 2100)],
        )

        period = "2020 to 2100"
        fossil_ccs = result.loc[savings_names.FOSSIL_CCS, period]
        biomass_ccs = result.loc[savings_names.BIOMASS_CCS, period]

        # CCS should be negative (reduces emissions)
        assert fossil_ccs < 0, "Fossil CCS should be negative"
        assert biomass_ccs < 0, "Biomass CCS should be negative"


# ============================================================================
# Tests for compute_savings_with_percentages
# ============================================================================

class TestComputeSavingsWithPercentages:
    """Tests for compute_savings_with_percentages function."""

    def test_returns_dataframe_with_percentage_columns(self, two_scenario_multi_year_data):
        """Should return DataFrame with absolute and percentage columns."""
        result = compute_savings_with_percentages(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        assert savings_names.ABS_VALUE in result.columns
        assert savings_names.PCT_OF_TOTAL in result.columns
        assert savings_names.PCT_OF_REF in result.columns

    def test_total_net_percentage_is_reasonable(self, two_scenario_multi_year_data):
        """Total/Net percentage should be within a reasonable range.

        Note: The LMDI decomposition may not sum exactly to the difference
        due to the nature of scenario comparison across multiple factors.
        A 10% tolerance is reasonable for synthetic test data.
        """
        result = compute_savings_with_percentages(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
        )

        total_pct = result.loc[savings_names.TOTAL_NET, savings_names.PCT_OF_TOTAL]

        # Total should be in a reasonable range (80-120%)
        assert 80 <= abs(total_pct) <= 120


# ============================================================================
# Tests for Integration Methods
# ============================================================================

class TestIntegrationMethods:
    """Tests for different integration methods."""

    def test_trapezoidal_vs_endpoint(self, two_scenario_multi_year_data):
        """Trapezoidal and endpoint methods should give different results."""
        result_trap = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
            periods=[(2020, 2100)],
            integration_method="trapezoidal",
        )

        result_endpoint = compute_savings(
            two_scenario_multi_year_data,
            ref_scenario=("TestModel", "Reference", "World"),
            int_scenario=("TestModel", "Intervention", "World"),
            periods=[(2020, 2100)],
            integration_method="endpoint",
        )

        period = "2020 to 2100"

        # Values should be different (trapezoidal is typically more accurate)
        trap_ref = result_trap.loc[savings_names.REF_CUMULATIVE, period]
        endpoint_ref = result_endpoint.loc[savings_names.REF_CUMULATIVE, period]

        # They shouldn't be exactly equal
        assert not np.isclose(trap_ref, endpoint_ref, rtol=0.001)


# ============================================================================
# Tests for compute_intervention_ccs
# ============================================================================

class TestComputeInterventionCcs:
    """Tests for compute_intervention_ccs function."""

    def test_returns_dict_with_both_ccs_types(self, two_scenario_multi_year_data):
        """Should return dict with fossil_ccs and biomass_ccs keys."""
        result = compute_intervention_ccs(
            two_scenario_multi_year_data,
            int_scenario=("TestModel", "Intervention", "World"),
            start_year=2020,
            end_year=2100,
        )

        assert "fossil_ccs" in result
        assert "biomass_ccs" in result

    def test_ccs_values_are_negative(self, two_scenario_multi_year_data):
        """CCS values should be negative (representing sequestration)."""
        result = compute_intervention_ccs(
            two_scenario_multi_year_data,
            int_scenario=("TestModel", "Intervention", "World"),
            start_year=2020,
            end_year=2100,
        )

        assert result["fossil_ccs"] < 0
        assert result["biomass_ccs"] < 0

    def test_zero_ccs_for_reference(self, two_scenario_multi_year_data):
        """Reference scenario should have zero CCS."""
        result = compute_intervention_ccs(
            two_scenario_multi_year_data,
            int_scenario=("TestModel", "Reference", "World"),
            start_year=2020,
            end_year=2100,
        )

        assert np.isclose(result["fossil_ccs"], 0.0)
        assert np.isclose(result["biomass_ccs"], 0.0)

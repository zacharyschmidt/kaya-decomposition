"""Tests for edge cases and error handling."""

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from kaya_decomposition import (
    compute_kaya_variables,
    compute_kaya_factors,
)
from kaya_decomposition.lmdi_cumulative import (
    compute_lmdi_cumulative,
    compute_lmdi_cumulative_sum,
    _logarithmic_mean,
)
from kaya_decomposition.all_sectors import (
    compute_all_sectors_lmdi_cumulative,
)


class TestEmptyDataHandling:
    """Tests for handling empty or missing data."""

    def test_compute_lmdi_cumulative_empty_data_raises_error(self):
        """Test that empty input raises ValueError."""
        # Create empty IamDataFrame
        empty_df = IamDataFrame(pd.DataFrame({
            "model": [],
            "scenario": [],
            "region": [],
            "variable": [],
            "unit": [],
            "year": [],
            "value": [],
        }))

        with pytest.raises(ValueError, match="Input data is empty"):
            compute_lmdi_cumulative(empty_df, base_year=2020)

    def test_compute_all_sectors_empty_data_raises_error(self):
        """Test that empty input raises ValueError for all_sectors."""
        empty_df = IamDataFrame(pd.DataFrame({
            "model": [],
            "scenario": [],
            "region": [],
            "variable": [],
            "unit": [],
            "year": [],
            "value": [],
        }))

        with pytest.raises(ValueError, match="Input data is empty"):
            compute_all_sectors_lmdi_cumulative(empty_df, base_year=2020)


class TestMissingBaseYear:
    """Tests for missing base year scenarios."""

    @pytest.fixture
    def data_without_base_year(self):
        """Create data that doesn't include 2020."""
        data = []
        for year in [2030, 2040, 2050]:  # No 2020!
            data.append({
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Total Fossil Carbon",
                "unit": "Mt CO2/yr",
                "year": year,
                "value": 10000 + year * 10,
            })
            data.append({
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Population",
                "unit": "million",
                "year": year,
                "value": 8000,
            })
            # Add other required variables
            for var in ["GNP/P", "FE/GNP", "PEDEq/FE", "PEFF/PEDEq", "TFC/PEFF"]:
                data.append({
                    "model": "Test",
                    "scenario": "Test",
                    "region": "World",
                    "variable": var,
                    "unit": "various",
                    "year": year,
                    "value": 1.0,
                })
        return IamDataFrame(pd.DataFrame(data))

    def test_missing_base_year_raises_error(self, data_without_base_year):
        """Test that missing base year raises informative error."""
        with pytest.raises(ValueError, match="Base year 2020 not found"):
            compute_lmdi_cumulative(data_without_base_year, base_year=2020)

    def test_error_message_lists_available_years(self, data_without_base_year):
        """Test that error message includes available years."""
        with pytest.raises(ValueError, match="Available years"):
            compute_lmdi_cumulative(data_without_base_year, base_year=2020)


class TestNaNHandling:
    """Tests for NaN value handling."""

    def test_logarithmic_mean_handles_nan(self):
        """Test that logarithmic mean handles NaN gracefully."""
        result = _logarithmic_mean(np.nan, 5)
        assert np.isnan(result)

        result = _logarithmic_mean(5, np.nan)
        assert np.isnan(result)

    def test_logarithmic_mean_handles_zero(self):
        """Test that logarithmic mean returns NaN for zero inputs."""
        result = _logarithmic_mean(0, 5)
        assert np.isnan(result)

        result = _logarithmic_mean(5, 0)
        assert np.isnan(result)

    def test_logarithmic_mean_handles_negative(self):
        """Test that logarithmic mean returns NaN for negative inputs."""
        result = _logarithmic_mean(-5, 10)
        assert np.isnan(result)

        result = _logarithmic_mean(10, -5)
        assert np.isnan(result)


class TestNegativeEmissions:
    """Tests for handling negative emission values (e.g., AFOLU sinks)."""

    @pytest.fixture
    def data_with_negative_afolu(self):
        """Create data with negative AFOLU emissions (carbon sink)."""
        data = []
        for year in [2020, 2030, 2050, 2100]:
            # Standard positive values for most variables
            data.extend([
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Population", "unit": "million", "year": year, "value": 8000},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "GDP|PPP", "unit": "billion USD", "year": year, "value": 100000},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Final Energy", "unit": "EJ/yr", "year": year, "value": 500},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Primary Energy", "unit": "EJ/yr", "year": year, "value": 600},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Primary Energy|Coal", "unit": "EJ/yr", "year": year, "value": 200},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Primary Energy|Oil", "unit": "EJ/yr", "year": year, "value": 200},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Primary Energy|Gas", "unit": "EJ/yr", "year": year, "value": 100},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Emissions|CO2|Energy and Industrial Processes", "unit": "Mt CO2/yr",
                 "year": year, "value": 35000},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Emissions|CO2|Industrial Processes", "unit": "Mt CO2/yr",
                 "year": year, "value": 2000},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Carbon Sequestration|CCS", "unit": "Mt CO2/yr",
                 "year": year, "value": 0},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Carbon Sequestration|CCS|Biomass", "unit": "Mt CO2/yr",
                 "year": year, "value": 0},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Carbon Sequestration|CCS|Fossil|Energy", "unit": "Mt CO2/yr",
                 "year": year, "value": 0},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Carbon Sequestration|CCS|Fossil|Industrial Processes", "unit": "Mt CO2/yr",
                 "year": year, "value": 0},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Carbon Sequestration|CCS|Biomass|Energy", "unit": "Mt CO2/yr",
                 "year": year, "value": 0},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Carbon Sequestration|CCS|Biomass|Industrial Processes", "unit": "Mt CO2/yr",
                 "year": year, "value": 0},
            ])

            # AFOLU becomes negative (net sink) in later years
            afolu_value = 5000 if year == 2020 else -500  # Negative in future
            data.append({
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": "Emissions|CO2|AFOLU", "unit": "Mt CO2/yr",
                "year": year, "value": afolu_value,
            })

        return IamDataFrame(pd.DataFrame(data))

    def test_negative_afolu_is_valid_input(self, data_with_negative_afolu):
        """Test that negative AFOLU values don't cause errors."""
        # Should compute without errors
        kaya_vars = compute_kaya_variables(data_with_negative_afolu)
        assert kaya_vars is not None

        factors = compute_kaya_factors(kaya_vars)
        assert factors is not None


class TestAllNegativeLmdiTerms:
    """Tests for the edge case where all LMDI terms are negative."""

    @pytest.fixture
    def declining_scenario_data(self):
        """Create scenario data where TFC decreases over time (all drivers negative)."""
        data = []
        for year in [2020, 2030, 2050]:
            # Decreasing trend in all factors
            factor = 1.0 - (year - 2020) * 0.005
            data.extend([
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Total Fossil Carbon", "unit": "Mt CO2/yr",
                 "year": year, "value": 40000 * factor},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "Population", "unit": "million",
                 "year": year, "value": 8000 * factor},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "GNP/P", "unit": "various",
                 "year": year, "value": 15 * factor},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "FE/GNP", "unit": "various",
                 "year": year, "value": 0.003 * factor},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "PEDEq/FE", "unit": "various",
                 "year": year, "value": 1.4 * factor},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "PEFF/PEDEq", "unit": "various",
                 "year": year, "value": 0.8 * factor},
                {"model": "Test", "scenario": "Test", "region": "World",
                 "variable": "TFC/PEFF", "unit": "various",
                 "year": year, "value": 72 * factor},
            ])
        return IamDataFrame(pd.DataFrame(data))

    def test_all_negative_terms_handled(self, declining_scenario_data):
        """Test that scenario with all negative LMDI terms doesn't crash."""
        result = compute_lmdi_cumulative(declining_scenario_data, base_year=2020)
        assert result is not None
        # The corrected values should sum to TFC difference
        for year in [2030, 2050]:
            year_data = result.filter(year=year).data
            # All values should be finite
            assert np.all(np.isfinite(year_data["value"]))


class TestDivisionByZeroProtection:
    """Tests for division by zero protection in LMDI calculations."""

    def test_logarithmic_mean_equal_values(self):
        """Test that L(a, a) = a (L'Hôpital's rule)."""
        result = _logarithmic_mean(5.0, 5.0)
        assert np.isclose(result, 5.0)

    def test_logarithmic_mean_nearly_equal_values(self):
        """Test that nearly equal values are handled correctly."""
        # Values that are very close but not exactly equal
        result = _logarithmic_mean(5.0, 5.0 + 1e-10)
        assert np.isfinite(result)
        assert np.isclose(result, 5.0, rtol=1e-6)

    def test_logarithmic_mean_array_with_equal_values(self):
        """Test array input where some values are equal."""
        a = np.array([2, 5, 10])
        b = np.array([8, 5, 20])  # Middle values are equal
        result = _logarithmic_mean(a, b)

        assert np.isfinite(result[0])  # 2 vs 8
        assert np.isclose(result[1], 5)  # 5 vs 5
        assert np.isfinite(result[2])  # 10 vs 20

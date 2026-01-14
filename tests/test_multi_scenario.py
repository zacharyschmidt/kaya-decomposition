"""Tests for handling multiple models, scenarios, and regions."""

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
)
from kaya_decomposition.all_sectors import (
    compute_all_sectors_lmdi_cumulative,
)


def _create_scenario_data(model, scenario, region, year_factor=1.0, value_factor=1.0):
    """Helper to create data for a single scenario."""
    data = []
    for year in [2020, 2030, 2050]:
        growth = 1.0 + (year - 2020) * 0.01 * year_factor
        base = 1000 * value_factor

        data.extend([
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Population", "unit": "million",
             "year": year, "value": 8000 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "GDP|PPP", "unit": "billion USD",
             "year": year, "value": 100000 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Final Energy", "unit": "EJ/yr",
             "year": year, "value": 500 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Primary Energy", "unit": "EJ/yr",
             "year": year, "value": 600 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Primary Energy|Coal", "unit": "EJ/yr",
             "year": year, "value": 200 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Primary Energy|Oil", "unit": "EJ/yr",
             "year": year, "value": 200 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Primary Energy|Gas", "unit": "EJ/yr",
             "year": year, "value": 100 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Emissions|CO2|Energy and Industrial Processes", "unit": "Mt CO2/yr",
             "year": year, "value": base * 35 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Emissions|CO2|Industrial Processes", "unit": "Mt CO2/yr",
             "year": year, "value": base * 2 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Emissions|CO2|AFOLU", "unit": "Mt CO2/yr",
             "year": year, "value": base * 5 * growth},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Carbon Sequestration|CCS", "unit": "Mt CO2/yr",
             "year": year, "value": 0},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Carbon Sequestration|CCS|Biomass", "unit": "Mt CO2/yr",
             "year": year, "value": 0},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Carbon Sequestration|CCS|Fossil|Energy", "unit": "Mt CO2/yr",
             "year": year, "value": 0},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Carbon Sequestration|CCS|Fossil|Industrial Processes", "unit": "Mt CO2/yr",
             "year": year, "value": 0},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Carbon Sequestration|CCS|Biomass|Energy", "unit": "Mt CO2/yr",
             "year": year, "value": 0},
            {"model": model, "scenario": scenario, "region": region,
             "variable": "Carbon Sequestration|CCS|Biomass|Industrial Processes", "unit": "Mt CO2/yr",
             "year": year, "value": 0},
        ])
    return data


class TestMultipleRegions:
    """Tests for handling multiple regions in the same DataFrame."""

    @pytest.fixture
    def multi_region_data(self):
        """Create data with multiple regions."""
        data = []
        data.extend(_create_scenario_data("Model1", "Baseline", "World", value_factor=1.0))
        data.extend(_create_scenario_data("Model1", "Baseline", "USA", value_factor=0.2))
        data.extend(_create_scenario_data("Model1", "Baseline", "Europe", value_factor=0.15))
        data.extend(_create_scenario_data("Model1", "Baseline", "China", value_factor=0.25))
        return IamDataFrame(pd.DataFrame(data))

    def test_compute_kaya_variables_multi_region(self, multi_region_data):
        """Test that Kaya variables are computed for all regions."""
        result = compute_kaya_variables(multi_region_data)

        assert result is not None
        regions = result.data["region"].unique()
        assert "World" in regions
        assert "USA" in regions
        assert "Europe" in regions
        assert "China" in regions

    def test_compute_kaya_factors_multi_region(self, multi_region_data):
        """Test that Kaya factors are computed for all regions."""
        kaya_vars = compute_kaya_variables(multi_region_data)
        result = compute_kaya_factors(kaya_vars)

        regions = result.data["region"].unique()
        assert len(regions) == 4

    def test_lmdi_cumulative_single_region_filter(self, multi_region_data):
        """Test that LMDI can filter to a single region."""
        kaya_vars = compute_kaya_variables(multi_region_data)
        factors = compute_kaya_factors(kaya_vars)

        result = compute_lmdi_cumulative(
            factors,
            base_year=2020,
            scenario=("Model1", "Baseline", "USA")
        )

        assert result is not None
        assert all(result.data["region"] == "USA")

    def test_lmdi_cumulative_default_uses_first(self, multi_region_data):
        """Test that default scenario uses first available."""
        kaya_vars = compute_kaya_variables(multi_region_data)
        factors = compute_kaya_factors(kaya_vars)

        result = compute_lmdi_cumulative(factors, base_year=2020)

        assert result is not None
        # Should use first scenario found
        assert len(result.data["region"].unique()) == 1


class TestMultipleModels:
    """Tests for handling multiple models in the same DataFrame."""

    @pytest.fixture
    def multi_model_data(self):
        """Create data with multiple models."""
        data = []
        data.extend(_create_scenario_data("IMAGE", "SSP2-Baseline", "World", year_factor=1.0))
        data.extend(_create_scenario_data("MESSAGE", "SSP2-Baseline", "World", year_factor=1.2))
        data.extend(_create_scenario_data("REMIND", "SSP2-Baseline", "World", year_factor=0.8))
        return IamDataFrame(pd.DataFrame(data))

    def test_compute_kaya_variables_multi_model(self, multi_model_data):
        """Test that Kaya variables are computed for all models."""
        result = compute_kaya_variables(multi_model_data)

        assert result is not None
        models = result.data["model"].unique()
        assert "IMAGE" in models
        assert "MESSAGE" in models
        assert "REMIND" in models

    def test_lmdi_cumulative_model_filter(self, multi_model_data):
        """Test that LMDI can filter to a specific model."""
        kaya_vars = compute_kaya_variables(multi_model_data)
        factors = compute_kaya_factors(kaya_vars)

        result = compute_lmdi_cumulative(
            factors,
            base_year=2020,
            scenario=("MESSAGE", "SSP2-Baseline", "World")
        )

        assert result is not None
        assert all(result.data["model"] == "MESSAGE")


class TestMultipleScenarios:
    """Tests for handling multiple scenarios in the same DataFrame."""

    @pytest.fixture
    def multi_scenario_data(self):
        """Create data with multiple scenarios from same model."""
        data = []
        data.extend(_create_scenario_data("Model1", "Baseline", "World", year_factor=1.0))
        data.extend(_create_scenario_data("Model1", "SSP1-19", "World", year_factor=0.5))
        data.extend(_create_scenario_data("Model1", "SSP5-85", "World", year_factor=2.0))
        return IamDataFrame(pd.DataFrame(data))

    def test_compute_kaya_variables_multi_scenario(self, multi_scenario_data):
        """Test that Kaya variables are computed for all scenarios."""
        result = compute_kaya_variables(multi_scenario_data)

        assert result is not None
        scenarios = result.data["scenario"].unique()
        assert "Baseline" in scenarios
        assert "SSP1-19" in scenarios
        assert "SSP5-85" in scenarios

    def test_lmdi_cumulative_scenario_filter(self, multi_scenario_data):
        """Test that LMDI can filter to a specific scenario."""
        kaya_vars = compute_kaya_variables(multi_scenario_data)
        factors = compute_kaya_factors(kaya_vars)

        result = compute_lmdi_cumulative(
            factors,
            base_year=2020,
            scenario=("Model1", "SSP1-19", "World")
        )

        assert result is not None
        assert all(result.data["scenario"] == "SSP1-19")


class TestCombinedMultiDimensional:
    """Tests for handling multiple dimensions simultaneously."""

    @pytest.fixture
    def complex_multi_data(self):
        """Create data with multiple models, scenarios, and regions."""
        data = []
        for model in ["Model1", "Model2"]:
            for scenario in ["Baseline", "Policy"]:
                for region in ["World", "Regional"]:
                    # Different factors for each combination
                    region_factor = 1.0 if region == "World" else 0.3
                    scenario_factor = 1.0 if scenario == "Baseline" else 0.7
                    model_factor = 1.0 if model == "Model1" else 1.2
                    combined_factor = region_factor * scenario_factor * model_factor
                    year_factor = 1.0 if scenario == "Baseline" else 0.5
                    data.extend(_create_scenario_data(
                        model, scenario, region,
                        value_factor=combined_factor,
                        year_factor=year_factor
                    ))
        return IamDataFrame(pd.DataFrame(data))

    def test_kaya_variables_all_combinations(self, complex_multi_data):
        """Test that all model/scenario/region combinations are processed."""
        result = compute_kaya_variables(complex_multi_data)

        assert result is not None
        # Should have data for all 2*2*2 = 8 combinations
        unique_combos = result.data.groupby(["model", "scenario", "region"]).size()
        assert len(unique_combos) == 8

    def test_lmdi_specific_combination(self, complex_multi_data):
        """Test LMDI for a specific combination."""
        kaya_vars = compute_kaya_variables(complex_multi_data)
        factors = compute_kaya_factors(kaya_vars)

        result = compute_lmdi_cumulative(
            factors,
            base_year=2020,
            scenario=("Model2", "Policy", "Regional")
        )

        assert result is not None
        assert all(result.data["model"] == "Model2")
        assert all(result.data["scenario"] == "Policy")
        assert all(result.data["region"] == "Regional")

    def test_results_differ_between_scenarios(self, complex_multi_data):
        """Test that different scenarios produce different results."""
        kaya_vars = compute_kaya_variables(complex_multi_data)
        factors = compute_kaya_factors(kaya_vars)

        result1 = compute_lmdi_cumulative(
            factors,
            base_year=2020,
            scenario=("Model1", "Baseline", "World")
        )
        result2 = compute_lmdi_cumulative(
            factors,
            base_year=2020,
            scenario=("Model1", "Policy", "World")
        )

        # Results should be valid for both
        assert result1 is not None
        assert result2 is not None

        # Get values at year 2050 for comparison
        val1 = result1.filter(year=2050).data["value"].sum()
        val2 = result2.filter(year=2050).data["value"].sum()

        # Values should differ due to different growth patterns
        # (They might be close in this test data, but not identical)
        assert not np.isclose(val1, val2, rtol=1e-10)

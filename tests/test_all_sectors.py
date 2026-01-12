"""Tests for all-sectors emissions analysis functions."""

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from kaya_decomposition import (
    input_variables,
    lmdi_cumulative as lmdi_names,
)
from kaya_decomposition.all_sectors import (
    compute_other_gases_emissions,
    compute_industrial_process_emissions,
    compute_land_use_emissions,
    compute_all_sectors_emissions,
    compute_all_sectors_lmdi_cumulative,
)


class TestComputeOtherGasesEmissions:
    """Tests for other gases emissions calculation."""

    def test_returns_iamdataframe(self, multi_year_all_sectors_dataframe):
        """Test that function returns an IamDataFrame."""
        result = compute_other_gases_emissions(multi_year_all_sectors_dataframe)
        assert isinstance(result, IamDataFrame)

    def test_gwp_conversion_ch4(self, multi_year_all_sectors_dataframe):
        """Test that CH4 GWP conversion is applied correctly."""
        result = compute_other_gases_emissions(multi_year_all_sectors_dataframe)

        # Get original CH4 value for base year
        ch4 = multi_year_all_sectors_dataframe.filter(
            variable=input_variables.EMISSIONS_CH4, year=2020
        ).data["value"].values[0]

        expected_ch4_co2eq = ch4 * input_variables.GWP_CH4

        # Result should include this contribution
        total = result.filter(year=2020).data["value"].values[0]
        # Total should be at least the CH4 contribution (plus N2O and F-gases)
        assert total >= expected_ch4_co2eq

    def test_output_variable_name(self, multi_year_all_sectors_dataframe):
        """Test that output has correct variable name."""
        result = compute_other_gases_emissions(multi_year_all_sectors_dataframe)
        assert "Emissions|Other Gases|CO2-equivalent" in result.variable

    def test_has_all_years(self, multi_year_all_sectors_dataframe):
        """Test that output has all input years."""
        result = compute_other_gases_emissions(multi_year_all_sectors_dataframe)
        input_years = set(multi_year_all_sectors_dataframe.data["year"].unique())
        output_years = set(result.data["year"].unique())
        assert input_years == output_years

    def test_invalid_fgas_method_raises_error(self, multi_year_all_sectors_dataframe):
        """Test that invalid fgas_method raises ValueError."""
        with pytest.raises(ValueError, match="fgas_method must be"):
            compute_other_gases_emissions(
                multi_year_all_sectors_dataframe,
                fgas_method="invalid"
            )

    def test_disaggregate_fgas_method(self):
        """Test disaggregated F-gas calculation."""
        # Create test data with individual F-gas species
        data = pd.DataFrame([
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|CH4",
                "unit": "Mt CH4/yr",
                "year": 2020,
                "value": 100,  # 100 Mt CH4
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|N2O",
                "unit": "kt N2O/yr",
                "year": 2020,
                "value": 1000,  # 1000 kt N2O
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|HFC",
                "unit": "kt HFC134a-equiv/yr",
                "year": 2020,
                "value": 100,  # 100 kt HFC134a-eq
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|PFC",
                "unit": "kt CF4-equiv/yr",
                "year": 2020,
                "value": 10,  # 10 kt CF4-eq
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|SF6",
                "unit": "kt SF6/yr",
                "year": 2020,
                "value": 5,  # 5 kt SF6
            },
        ])
        test_df = IamDataFrame(data)

        result = compute_other_gases_emissions(test_df, fgas_method="disaggregate")
        total = result.filter(year=2020).data["value"].values[0]

        # Expected calculation:
        # CH4: 100 Mt × 27.9 = 2790 Mt CO2-eq
        # N2O: 1000 kt × 273 / 1000 = 273 Mt CO2-eq
        # HFC: 100 kt × 1530 / 1000 = 153 Mt CO2-eq
        # PFC: 10 kt × 7380 / 1000 = 73.8 Mt CO2-eq
        # SF6: 5 kt × 25200 / 1000 = 126 Mt CO2-eq
        # Total = 2790 + 273 + 153 + 73.8 + 126 = 3415.8 Mt CO2-eq
        expected = (
            100 * input_variables.GWP_CH4 +      # CH4
            1000 * input_variables.GWP_N2O / 1000 +  # N2O
            100 * input_variables.GWP_HFC134A / 1000 +  # HFC
            10 * input_variables.GWP_CF4 / 1000 +    # PFC
            5 * input_variables.GWP_SF6 / 1000       # SF6
        )

        assert np.isclose(total, expected, rtol=1e-6), \
            f"Expected {expected:.1f}, got {total:.1f}"

    def test_aggregate_matches_when_fgases_precomputed(self):
        """Test that aggregate method uses pre-aggregated F-gases."""
        # Create test data with pre-aggregated F-gases
        data = pd.DataFrame([
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|CH4",
                "unit": "Mt CH4/yr",
                "year": 2020,
                "value": 100,
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|N2O",
                "unit": "kt N2O/yr",
                "year": 2020,
                "value": 1000,
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|F-Gases",
                "unit": "Mt CO2-equiv/yr",
                "year": 2020,
                "value": 500,  # Pre-aggregated F-gases in CO2-eq
            },
        ])
        test_df = IamDataFrame(data)

        result = compute_other_gases_emissions(test_df, fgas_method="aggregate")
        total = result.filter(year=2020).data["value"].values[0]

        # Expected calculation:
        # CH4: 100 Mt × 27.9 = 2790 Mt CO2-eq
        # N2O: 1000 kt × 273 / 1000 = 273 Mt CO2-eq
        # F-gases: 500 Mt CO2-eq (already aggregated)
        # Total = 2790 + 273 + 500 = 3563 Mt CO2-eq
        expected = (
            100 * input_variables.GWP_CH4 +
            1000 * input_variables.GWP_N2O / 1000 +
            500  # Pre-aggregated F-gases
        )

        assert np.isclose(total, expected, rtol=1e-6), \
            f"Expected {expected:.1f}, got {total:.1f}"


class TestComputeIndustrialProcessEmissions:
    """Tests for industrial process emissions calculation."""

    def test_returns_iamdataframe(self, multi_year_all_sectors_dataframe):
        """Test that function returns an IamDataFrame."""
        result = compute_industrial_process_emissions(multi_year_all_sectors_dataframe)
        assert isinstance(result, IamDataFrame)

    def test_output_variable_name(self, multi_year_all_sectors_dataframe):
        """Test that output has correct variable name."""
        result = compute_industrial_process_emissions(multi_year_all_sectors_dataframe)
        assert "Net Industrial Carbon" in result.variable

    def test_subtracts_ccs(self):
        """Test that CCS is subtracted from gross emissions."""
        # Create test data with CCS
        data = pd.DataFrame([
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|CO2|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 1000,
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Carbon Sequestration|CCS|Fossil|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 200,
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Carbon Sequestration|CCS|Biomass|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 100,
            },
        ])
        test_df = IamDataFrame(data)

        result = compute_industrial_process_emissions(test_df)
        nic = result.filter(year=2020).data["value"].values[0]

        # Net Industrial Carbon = 1000 - 200 - 100 = 700
        assert np.isclose(nic, 700)


class TestComputeTotalIndustrialCarbon:
    """Tests for total industrial carbon (TIC) calculation."""

    def test_returns_iamdataframe(self, multi_year_all_sectors_dataframe):
        """Test that function returns an IamDataFrame."""
        from kaya_decomposition.all_sectors import compute_total_industrial_carbon
        result = compute_total_industrial_carbon(multi_year_all_sectors_dataframe)
        assert isinstance(result, IamDataFrame)

    def test_output_variable_name(self, multi_year_all_sectors_dataframe):
        """Test that output has correct variable name."""
        from kaya_decomposition.all_sectors import compute_total_industrial_carbon
        result = compute_total_industrial_carbon(multi_year_all_sectors_dataframe)
        assert "Total Industrial Carbon" in result.variable

    def test_adds_ccs_to_gross(self):
        """Test that CCS is added back to get gross emissions."""
        from kaya_decomposition.all_sectors import compute_total_industrial_carbon
        # Create test data with CCS
        data = pd.DataFrame([
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|CO2|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 1000,
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Carbon Sequestration|CCS|Fossil|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 200,
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Carbon Sequestration|CCS|Biomass|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 100,
            },
        ])
        test_df = IamDataFrame(data)

        result = compute_total_industrial_carbon(test_df)
        tic = result.filter(year=2020).data["value"].values[0]

        # Total Industrial Carbon = 1000 + 200 + 100 = 1300
        assert np.isclose(tic, 1300)

    def test_tic_equals_nic_when_no_ccs(self, multi_year_all_sectors_dataframe):
        """Test that TIC equals NIC when there's no CCS."""
        from kaya_decomposition.all_sectors import compute_total_industrial_carbon
        # The multi_year fixture has no CCS, so TIC should equal NIC
        tic_result = compute_total_industrial_carbon(multi_year_all_sectors_dataframe)
        nic_result = compute_industrial_process_emissions(multi_year_all_sectors_dataframe)

        for year in [2020, 2030, 2040, 2050]:
            tic = tic_result.filter(year=year).data["value"].values[0]
            nic = nic_result.filter(year=year).data["value"].values[0]
            assert np.isclose(tic, nic), \
                f"Year {year}: TIC {tic} should equal NIC {nic} when no CCS"

    def test_tic_minus_nic_equals_twice_ccs(self):
        """Test that TIC - NIC = 2 * CCS.

        Since TIC = IP + CCS and NIC = IP - CCS, their difference is 2*CCS.
        """
        from kaya_decomposition.all_sectors import compute_total_industrial_carbon
        # Create test data with CCS
        data = pd.DataFrame([
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Emissions|CO2|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 1000,
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Carbon Sequestration|CCS|Fossil|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 200,
            },
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Carbon Sequestration|CCS|Biomass|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 100,
            },
        ])
        test_df = IamDataFrame(data)

        tic_result = compute_total_industrial_carbon(test_df)
        nic_result = compute_industrial_process_emissions(test_df)

        tic = tic_result.filter(year=2020).data["value"].values[0]
        nic = nic_result.filter(year=2020).data["value"].values[0]

        # TIC - NIC = 2 * total CCS
        total_ccs = 200 + 100  # fossil + biomass CCS
        assert np.isclose(tic - nic, 2 * total_ccs), \
            f"TIC ({tic}) - NIC ({nic}) should equal 2*CCS ({2*total_ccs})"


class TestComputeLandUseEmissions:
    """Tests for land use emissions calculation."""

    def test_returns_iamdataframe(self, multi_year_all_sectors_dataframe):
        """Test that function returns an IamDataFrame."""
        result = compute_land_use_emissions(multi_year_all_sectors_dataframe)
        assert isinstance(result, IamDataFrame)

    def test_output_variable_name(self, multi_year_all_sectors_dataframe):
        """Test that output has correct variable name."""
        result = compute_land_use_emissions(multi_year_all_sectors_dataframe)
        assert "Emissions|CO2|Land Use" in result.variable

    def test_values_match_afolu(self, multi_year_all_sectors_dataframe):
        """Test that values match AFOLU input."""
        result = compute_land_use_emissions(multi_year_all_sectors_dataframe)

        for year in [2020, 2030, 2040, 2050]:
            afolu = multi_year_all_sectors_dataframe.filter(
                variable=input_variables.EMISSIONS_CO2_AFOLU, year=year
            ).data["value"].values[0]

            land_use = result.filter(year=year).data["value"].values[0]

            assert np.isclose(afolu, land_use)


class TestComputeAllSectorsEmissions:
    """Tests for all-sectors emissions calculation."""

    def test_returns_iamdataframe(self, multi_year_all_sectors_dataframe):
        """Test that function returns an IamDataFrame."""
        result = compute_all_sectors_emissions(multi_year_all_sectors_dataframe)
        assert isinstance(result, IamDataFrame)

    def test_has_all_components(self, multi_year_all_sectors_dataframe):
        """Test that output has all expected components."""
        result = compute_all_sectors_emissions(multi_year_all_sectors_dataframe)

        expected_vars = [
            "Net Fossil Carbon",
            "Net Industrial Carbon",
            "Emissions|Other Gases|CO2-equivalent",
            "Emissions|CO2|Land Use",
        ]

        for var in expected_vars:
            assert var in result.variable, f"Missing variable: {var}"


class TestComputeAllSectorsLmdiCumulative:
    """Tests for the complete all-sectors LMDI analysis."""

    def test_returns_dataframe(self, multi_year_all_sectors_dataframe):
        """Test that function returns a pandas DataFrame."""
        result = compute_all_sectors_lmdi_cumulative(
            multi_year_all_sectors_dataframe,
            base_year=2020,
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_all_rows(self, multi_year_all_sectors_dataframe):
        """Test that output has all expected row labels."""
        result = compute_all_sectors_lmdi_cumulative(
            multi_year_all_sectors_dataframe,
            base_year=2020,
        )

        expected_rows = [
            lmdi_names.Pop_cumulative,
            lmdi_names.GNP_per_P_cumulative,
            lmdi_names.FE_per_GNP_cumulative,
            lmdi_names.PEdeq_per_FE_cumulative,
            lmdi_names.PEFF_per_PEDEq_cumulative,
            lmdi_names.TFC_per_PEFF_cumulative,
            lmdi_names.Industrial_Process,
            lmdi_names.Other_Gases,
            lmdi_names.Land_Use,
            lmdi_names.Total_Net_Emissions,
        ]

        for row in expected_rows:
            assert row in result.index, f"Missing row: {row}"

    def test_total_equals_sum_of_components(self, multi_year_all_sectors_dataframe):
        """Test that Total Net Emissions equals sum of all components."""
        result = compute_all_sectors_lmdi_cumulative(
            multi_year_all_sectors_dataframe,
            base_year=2020,
        )

        for col in result.columns:
            component_sum = result.loc[
                result.index != lmdi_names.Total_Net_Emissions, col
            ].sum()
            total = result.loc[lmdi_names.Total_Net_Emissions, col]

            assert np.isclose(component_sum, total, rtol=0.01), \
                f"Period {col}: sum={component_sum}, total={total}"

    def test_custom_periods(self, multi_year_all_sectors_dataframe):
        """Test with custom period specification."""
        result = compute_all_sectors_lmdi_cumulative(
            multi_year_all_sectors_dataframe,
            base_year=2020,
            periods=[(2020, 2030), (2030, 2050)],
        )

        assert "2020 to 2030" in result.columns
        assert "2030 to 2050" in result.columns

    def test_scenario_filter(self, multi_year_all_sectors_dataframe):
        """Test filtering to specific scenario."""
        result = compute_all_sectors_lmdi_cumulative(
            multi_year_all_sectors_dataframe,
            base_year=2020,
            scenario=("TestModel", "TestScenario", "World"),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_row_order(self, multi_year_all_sectors_dataframe):
        """Test that rows are in the expected order."""
        result = compute_all_sectors_lmdi_cumulative(
            multi_year_all_sectors_dataframe,
            base_year=2020,
        )

        expected_order = [
            lmdi_names.Pop_cumulative,
            lmdi_names.GNP_per_P_cumulative,
            lmdi_names.FE_per_GNP_cumulative,
            lmdi_names.PEdeq_per_FE_cumulative,
            lmdi_names.PEFF_per_PEDEq_cumulative,
            lmdi_names.TFC_per_PEFF_cumulative,
            lmdi_names.Industrial_Process,
            lmdi_names.Other_Gases,
            lmdi_names.Land_Use,
            lmdi_names.Total_Net_Emissions,
        ]

        actual_order = list(result.index)
        assert actual_order == expected_order

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

    def test_adds_back_biomass_ccs(self):
        """Test that biomass-industrial CCS is added back, fossil is not re-subtracted."""
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

        # Reported IP (1000) is already net of fossil-industrial CCS (200), so
        # fossil CCS is NOT subtracted again; biomass-industrial CCS (100) is
        # added back to strip its biogenic credit out of the fossil chain.
        # NIC = 1000 + 100 = 1100.
        assert np.isclose(nic, 1100)

    def test_nic_adds_back_biomass_ccs(self):
        """NIC adds back biomass-industrial CCS (input held fixed).

        The reported industrial-process CO2 is net of biomass-industrial CCS.
        NIC adds that credit back so it nets out only *fossil* sequestration;
        the biogenic removal is accounted once, separately, via CDR. So raising
        biomass-industrial CCS by delta raises NIC by exactly delta.
        """
        base_rows = [
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
        ]
        baseline_df = IamDataFrame(pd.DataFrame(base_rows))
        high_biomass_df = IamDataFrame(pd.DataFrame(base_rows + [
            {
                "model": "Test",
                "scenario": "Test",
                "region": "World",
                "variable": "Carbon Sequestration|CCS|Biomass|Industrial Processes",
                "unit": "Mt CO2/yr",
                "year": 2020,
                "value": 500,
            },
        ]))

        nic_baseline = compute_industrial_process_emissions(
            baseline_df
        ).filter(year=2020).data["value"].values[0]
        nic_high_biomass = compute_industrial_process_emissions(
            high_biomass_df
        ).filter(year=2020).data["value"].values[0]

        # baseline has no biomass-industrial CCS (NIC = 1000); the variant adds
        # 500, which is added back into NIC (fossil CCS is never re-subtracted).
        assert np.isclose(nic_baseline, 1000)
        assert np.isclose(nic_high_biomass, nic_baseline + 500)


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

    def test_tic_minus_nic_equals_fossil_ccs(self):
        """Test that TIC - NIC = fossil industrial CCS.

        TIC = IP + fossil_CCS + biomass_CCS (gross, all captured carbon added
        back). NIC = IP + biomass_CCS (fossil CCS already netted in the input;
        biomass added back). Their difference is therefore fossil_CCS.
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

        # TIC - NIC = fossil industrial CCS
        fossil_ccs = 200
        assert np.isclose(tic - nic, fossil_ccs), \
            f"TIC ({tic}) - NIC ({nic}) should equal fossil_CCS ({fossil_ccs})"


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


class TestComputeTotalCdr:
    """Tests for total CDR (carbon dioxide removal) calculation."""

    def test_returns_iamdataframe(self, multi_year_all_sectors_dataframe):
        """Test that function returns an IamDataFrame."""
        from kaya_decomposition.all_sectors import compute_total_cdr
        result = compute_total_cdr(multi_year_all_sectors_dataframe)
        assert isinstance(result, IamDataFrame)

    def test_output_variable_name(self, multi_year_all_sectors_dataframe):
        """Test that output has correct variable name."""
        from kaya_decomposition.all_sectors import compute_total_cdr
        result = compute_total_cdr(multi_year_all_sectors_dataframe)
        assert "Carbon Dioxide Removal" in result.variable

    def test_zero_when_no_removal_reported(self, multi_year_all_sectors_dataframe):
        """Test that CDR is zero when no removal variables are present."""
        from kaya_decomposition.all_sectors import compute_total_cdr
        # multi_year_all_sectors_dataframe has CCS == 0 everywhere and no
        # Carbon Removal|... variables at all.
        result = compute_total_cdr(multi_year_all_sectors_dataframe)
        assert (result.data["value"] == 0).all()

    def test_legacy_biomass_ccs_fallback(self):
        """Test that legacy CCS|Biomass variables are used when no modern
        Carbon Removal variable is reported."""
        from kaya_decomposition.all_sectors import compute_total_cdr
        data = pd.DataFrame([
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": "Carbon Sequestration|CCS|Biomass|Energy",
                "unit": "Mt CO2/yr", "year": 2020, "value": 40,
            },
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": "Carbon Sequestration|CCS|Biomass|Industrial Processes",
                "unit": "Mt CO2/yr", "year": 2020, "value": 10,
            },
        ])
        result = compute_total_cdr(IamDataFrame(data))
        cdr = result.filter(year=2020).data["value"].values[0]

        # Removal is reported as positive sequestration; CDR is negative.
        assert np.isclose(cdr, -50)

    def test_prefers_modern_biomass_variable_over_legacy(self):
        """Test that the modern Carbon Removal variable wins over the
        legacy CCS split, and the two are never summed together."""
        from kaya_decomposition.all_sectors import compute_total_cdr
        data = pd.DataFrame([
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": input_variables.CARBON_REMOVAL_BIOMASS,
                "unit": "Mt CO2/yr", "year": 2020, "value": 70,
            },
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": "Carbon Sequestration|CCS|Biomass|Energy",
                "unit": "Mt CO2/yr", "year": 2020, "value": 40,
            },
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": "Carbon Sequestration|CCS|Biomass|Industrial Processes",
                "unit": "Mt CO2/yr", "year": 2020, "value": 10,
            },
        ])
        result = compute_total_cdr(IamDataFrame(data))
        cdr = result.filter(year=2020).data["value"].values[0]

        assert np.isclose(cdr, -70)

    def test_includes_daccs_and_land_use(self):
        """Test that DACCS and land-based removal are included in the total."""
        from kaya_decomposition.all_sectors import compute_total_cdr
        data = pd.DataFrame([
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": input_variables.CARBON_REMOVAL_BIOMASS,
                "unit": "Mt CO2/yr", "year": 2020, "value": 70,
            },
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": input_variables.CARBON_REMOVAL_DACCS,
                "unit": "Mt CO2/yr", "year": 2020, "value": 20,
            },
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": input_variables.CARBON_REMOVAL_LAND_USE,
                "unit": "Mt CO2/yr", "year": 2020, "value": 5,
            },
        ])
        result = compute_total_cdr(IamDataFrame(data))
        cdr = result.filter(year=2020).data["value"].values[0]

        assert np.isclose(cdr, -95)

    def test_excludes_fossil_ccs(self):
        """Test that fossil CCS (abatement, not removal) is never included."""
        from kaya_decomposition.all_sectors import compute_total_cdr
        data = pd.DataFrame([
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": "Carbon Sequestration|CCS|Fossil|Energy",
                "unit": "Mt CO2/yr", "year": 2020, "value": 200,
            },
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": "Carbon Sequestration|CCS|Fossil|Industrial Processes",
                "unit": "Mt CO2/yr", "year": 2020, "value": 100,
            },
        ])
        result = compute_total_cdr(IamDataFrame(data))
        cdr = result.filter(year=2020).data["value"].values[0]

        assert np.isclose(cdr, 0)

    def test_unit_is_mt_co2(self):
        """Test that output unit is Mt CO2/yr."""
        from kaya_decomposition.all_sectors import compute_total_cdr
        data = pd.DataFrame([
            {
                "model": "Test", "scenario": "Test", "region": "World",
                "variable": input_variables.CARBON_REMOVAL_DACCS,
                "unit": "Mt CO2/yr", "year": 2020, "value": 20,
            },
        ])
        result = compute_total_cdr(IamDataFrame(data))
        assert set(result.unit) == {"Mt CO2/yr"}


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
            "Carbon Dioxide Removal",
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
            lmdi_names.Fossil_Energy_CCS,
            lmdi_names.Total_CDR,
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
            lmdi_names.Fossil_Energy_CCS,
            lmdi_names.Total_CDR,
            lmdi_names.Total_Net_Emissions,
        ]

        actual_order = list(result.index)
        assert actual_order == expected_order


class TestAllSectorsNetInvariant:
    """Regression tests pinning the correct net-emissions accounting.

    These would have caught the biomass CDR double-counting, the fossil-industrial
    CCS double-subtraction, and the cumulative fossil-energy CCS gap. They are the
    load-bearing tests because every Excel fixture has CCS = 0.
    """

    def _levels_sum_equals_reported_net(self, df):
        result = compute_all_sectors_emissions(df)
        rdata = result.data
        idata = df.data

        for year in sorted(idata["year"].unique()):
            def rget(var):
                v = rdata[(rdata["variable"] == var) & (rdata["year"] == year)]["value"]
                return v.values[0] if len(v) else 0.0

            def iget(var):
                v = idata[(idata["variable"] == var) & (idata["year"] == year)]["value"]
                return v.values[0] if len(v) else 0.0

            co2_sum = (
                rget("Net Fossil Carbon")
                + rget("Net Industrial Carbon")
                + rget("Emissions|CO2|Land Use")
                + rget("Carbon Dioxide Removal")
            )
            expected = (
                iget(input_variables.EMISSIONS_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES)
                + iget(input_variables.EMISSIONS_CO2_AFOLU)
            )
            assert np.isclose(co2_sum, expected, rtol=1e-9), (
                f"Year {year}: NFC+NIC+LU+CDR={co2_sum} != EIP+AFOLU={expected}"
            )

    def test_levels_invariant_single_year(self, test_dataframe):
        """NFC + NIC + Land Use + CDR == EIP + AFOLU (single year, nonzero CCS)."""
        self._levels_sum_equals_reported_net(test_dataframe)

    def test_levels_invariant_deep_mitigation(self, multi_year_beccs_dataframe):
        """Same identity across a multi-year path with nonzero fossil AND biomass CCS."""
        self._levels_sum_equals_reported_net(multi_year_beccs_dataframe)

    def test_cumulative_total_matches_true_net(self, multi_year_beccs_dataframe):
        """Cumulative Total Net Emissions reconstructs the true net change.

        Independent (non-circular) check: compare the table's Total Net Emissions
        to the trapezoidally-integrated change in true net emissions
        (EIP + AFOLU + Other Gases) computed straight from the raw inputs. This
        exercises the gross-TFC Kaya chain plus the Fossil Energy CCS and CDR rows
        together, on a CCS-heavy scenario.
        """
        from kaya_decomposition.utils import trapezoidal_integrate

        base_year = 2020
        result = compute_all_sectors_lmdi_cumulative(
            multi_year_beccs_dataframe, base_year=base_year, periods=[(2020, 2050)],
        )

        # Fossil Energy CCS row is present and negative (abatement) for this fixture.
        assert result.loc[lmdi_names.Fossil_Energy_CCS, "2020 to 2050"] < 0

        idata = multi_year_beccs_dataframe.data
        og = compute_other_gases_emissions(multi_year_beccs_dataframe).data
        years = sorted(idata["year"].unique())

        def iget(var, year):
            v = idata[(idata["variable"] == var) & (idata["year"] == year)]["value"]
            return v.values[0] if len(v) else 0.0

        def ogget(year):
            v = og[og["year"] == year]["value"]
            return v.values[0] if len(v) else 0.0

        net = {
            y: iget(input_variables.EMISSIONS_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES, y)
            + iget(input_variables.EMISSIONS_CO2_AFOLU, y)
            + ogget(y)
            for y in years
        }
        diff = pd.DataFrame(
            [{"year": y, "value": net[y] - net[base_year]} for y in years]
        ).sort_values("year")
        expected = trapezoidal_integrate(diff, 2020, 2050) / 1000

        total = result.loc[lmdi_names.Total_Net_Emissions, "2020 to 2050"]
        assert np.isclose(total, expected, rtol=1e-6), (
            f"Total Net Emissions {total} != independent true net {expected}"
        )

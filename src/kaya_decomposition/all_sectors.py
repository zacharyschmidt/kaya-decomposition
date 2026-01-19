"""Compute all-sectors emissions analysis including non-CO2 and land use."""

import numpy as np
import pandas as pd

import pyam
from kaya_decomposition.constants import (
    input_variables,
    kaya_variables as kaya_variable_names,
    lmdi_cumulative as lmdi_cumulative_names,
)
from kaya_decomposition.variables import compute_kaya_variables
from kaya_decomposition.factors import compute_kaya_factors
from kaya_decomposition.lmdi_cumulative import (
    compute_lmdi_cumulative_sum,
)
from kaya_decomposition.utils import trapezoidal_integrate


def compute_other_gases_emissions(input_data, fgas_method="aggregate", missing_value=0.0):
    """Compute total non-CO2 greenhouse gas emissions in CO2-equivalent.

    Converts CH4, N2O, and F-gases to CO2-equivalent using AR6 GWP values.

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Input data containing Emissions|CH4, Emissions|N2O, and F-gas variables.
    fgas_method : str, optional
        Method for computing F-gas contribution:
        - "aggregate" (default): Use pre-aggregated Emissions|F-Gases variable
          (assumes already in CO2-equivalent Mt/yr)
        - "disaggregate": Compute from individual HFC, PFC, SF6 emissions
          using their specific GWP values
    missing_value : float, optional
        Value to use when input data is missing. Default is 0.0.
        Use np.nan to propagate missing data as NaN.

    Returns
    -------
    pyam.IamDataFrame
        Total other gases emissions in Mt CO2-equivalent/yr.
        Variable name: "Emissions|Other Gases|CO2-equivalent"

    Notes
    -----
    When using fgas_method="disaggregate", the function expects:
    - Emissions|HFC in kt HFC134a-equivalent/yr
    - Emissions|PFC in kt CF4-equivalent/yr
    - Emissions|SF6 in kt SF6/yr

    The GWP values used are from IPCC AR6:
    - HFC134a: 1530
    - CF4 (PFC): 7380
    - SF6: 25200

    The disaggregate method matches the Excel OtherGases sheet calculation.
    """
    if fgas_method not in ("aggregate", "disaggregate"):
        raise ValueError(
            f"fgas_method must be 'aggregate' or 'disaggregate', got '{fgas_method}'"
        )

    data = input_data.data
    result_rows = []

    # Get unique model/scenario/region/year combinations
    groupby_cols = ["model", "scenario", "region", "year"]
    groups = data.groupby(groupby_cols)

    for group_key, group_data in groups:
        model, scenario, region, year = group_key

        # Get CH4 emissions and convert to CO2-equivalent
        ch4_data = group_data[group_data["variable"] == input_variables.EMISSIONS_CH4]
        ch4_co2eq = missing_value
        if len(ch4_data) > 0:
            # CH4 in Mt/yr, multiply by GWP to get Mt CO2-eq/yr
            ch4_co2eq = ch4_data["value"].values[0] * input_variables.GWP_CH4

        # Get N2O emissions and convert to CO2-equivalent
        n2o_data = group_data[group_data["variable"] == input_variables.EMISSIONS_N2O]
        n2o_co2eq = missing_value
        if len(n2o_data) > 0:
            # N2O in kt/yr, convert to Mt and multiply by GWP
            n2o_co2eq = n2o_data["value"].values[0] * input_variables.GWP_N2O / 1000

        # Compute F-gases based on method
        if fgas_method == "aggregate":
            # Get F-gases (already in CO2-equivalent Mt/yr)
            fgases_data = group_data[
                group_data["variable"] == input_variables.EMISSIONS_FGASES
            ]
            fgases_co2eq = missing_value
            if len(fgases_data) > 0:
                fgases_co2eq = fgases_data["value"].values[0]
        else:
            # Compute from individual F-gas species
            # HFC: kt HFC134a-eq → Mt CO2-eq (GWP = 1530, divide by 1000 for kt→Mt)
            hfc_data = group_data[
                group_data["variable"] == input_variables.EMISSIONS_HFC
            ]
            hfc_co2eq = missing_value
            if len(hfc_data) > 0:
                hfc_co2eq = hfc_data["value"].values[0] * input_variables.GWP_HFC134A / 1000

            # PFC: kt CF4-eq → Mt CO2-eq (GWP = 7380)
            pfc_data = group_data[
                group_data["variable"] == input_variables.EMISSIONS_PFC
            ]
            pfc_co2eq = missing_value
            if len(pfc_data) > 0:
                pfc_co2eq = pfc_data["value"].values[0] * input_variables.GWP_CF4 / 1000

            # SF6: kt SF6 → Mt CO2-eq (GWP = 25200)
            sf6_data = group_data[
                group_data["variable"] == input_variables.EMISSIONS_SF6
            ]
            sf6_co2eq = missing_value
            if len(sf6_data) > 0:
                sf6_co2eq = sf6_data["value"].values[0] * input_variables.GWP_SF6 / 1000

            fgases_co2eq = hfc_co2eq + pfc_co2eq + sf6_co2eq

        # Total other gases
        total_other_gases = ch4_co2eq + n2o_co2eq + fgases_co2eq

        result_rows.append({
            "model": model,
            "scenario": scenario,
            "region": region,
            "variable": "Emissions|Other Gases|CO2-equivalent",
            "unit": "Mt CO2-equiv/yr",
            "year": year,
            "value": total_other_gases,
        })

    return pyam.IamDataFrame(pd.DataFrame(result_rows))


def compute_industrial_process_emissions(input_data, missing_value=0.0):
    """Compute net industrial process carbon emissions.

    This is the industrial process emissions minus any industrial CCS.

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Input data with Industrial Processes emissions and CCS.
    missing_value : float, optional
        Value to use when input data is missing. Default is 0.0.
        Use np.nan to propagate missing data as NaN.

    Returns
    -------
    pyam.IamDataFrame
        Net industrial process carbon (NIC) in Mt CO2/yr.
        Variable name: "Net Industrial Carbon"
    """
    data = input_data.data
    result_rows = []

    # Get unique model/scenario/region/year combinations
    groupby_cols = ["model", "scenario", "region", "year"]
    groups = data.groupby(groupby_cols)

    for group_key, group_data in groups:
        model, scenario, region, year = group_key

        # Get industrial process emissions
        ip_data = group_data[
            group_data["variable"] == input_variables.EMISSIONS_CO2_INDUSTRIAL_PROCESSES
        ]
        ip_emissions = missing_value
        if len(ip_data) > 0:
            ip_emissions = ip_data["value"].values[0]

        # Get CCS from fossil industrial processes
        ccs_fossil_ip = group_data[
            group_data["variable"] == input_variables.CCS_FOSSIL_INDUSTRY
        ]
        ccs_fossil = 0.0
        if len(ccs_fossil_ip) > 0:
            ccs_fossil = ccs_fossil_ip["value"].values[0]

        # Get CCS from biomass industrial processes
        ccs_biomass_ip = group_data[
            group_data["variable"] == input_variables.CCS_BIOMASS_INDUSTRY
        ]
        ccs_biomass = 0.0
        if len(ccs_biomass_ip) > 0:
            ccs_biomass = ccs_biomass_ip["value"].values[0]

        # Net industrial carbon = gross emissions - CCS
        nic = ip_emissions - ccs_fossil - ccs_biomass

        result_rows.append({
            "model": model,
            "scenario": scenario,
            "region": region,
            "variable": "Net Industrial Carbon",
            "unit": "Mt CO2/yr",
            "year": year,
            "value": nic,
        })

    return pyam.IamDataFrame(pd.DataFrame(result_rows))


def compute_total_industrial_carbon(input_data):
    """Compute total (gross) industrial process carbon emissions.

    This is the industrial process emissions before any CCS is applied,
    representing the total industrial carbon that would be emitted without
    carbon capture.

    TIC = IP_emissions + CCS_fossil_IP + CCS_biomass_IP

    Note: CCS variables represent carbon that was captured, so adding them
    back to net emissions gives the gross total.

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Input data with Industrial Processes emissions and CCS.

    Returns
    -------
    pyam.IamDataFrame
        Total industrial process carbon (TIC) in Mt CO2/yr.
        Variable name: "Total Industrial Carbon"

    See Also
    --------
    compute_industrial_process_emissions : Computes net industrial carbon (after CCS)
    """
    data = input_data.data
    result_rows = []

    # Get unique model/scenario/region/year combinations
    groupby_cols = ["model", "scenario", "region", "year"]
    groups = data.groupby(groupby_cols)

    for group_key, group_data in groups:
        model, scenario, region, year = group_key

        # Get industrial process emissions (net, as reported)
        ip_data = group_data[
            group_data["variable"] == input_variables.EMISSIONS_CO2_INDUSTRIAL_PROCESSES
        ]
        ip_emissions = 0.0
        if len(ip_data) > 0:
            ip_emissions = ip_data["value"].values[0]

        # Get CCS from fossil industrial processes
        ccs_fossil_ip = group_data[
            group_data["variable"] == input_variables.CCS_FOSSIL_INDUSTRY
        ]
        ccs_fossil = 0.0
        if len(ccs_fossil_ip) > 0:
            ccs_fossil = ccs_fossil_ip["value"].values[0]

        # Get CCS from biomass industrial processes
        ccs_biomass_ip = group_data[
            group_data["variable"] == input_variables.CCS_BIOMASS_INDUSTRY
        ]
        ccs_biomass = 0.0
        if len(ccs_biomass_ip) > 0:
            ccs_biomass = ccs_biomass_ip["value"].values[0]

        # Total industrial carbon = net emissions + all CCS
        # This represents gross emissions before any carbon capture
        tic = ip_emissions + ccs_fossil + ccs_biomass

        result_rows.append({
            "model": model,
            "scenario": scenario,
            "region": region,
            "variable": "Total Industrial Carbon",
            "unit": "Mt CO2/yr",
            "year": year,
            "value": tic,
        })

    return pyam.IamDataFrame(pd.DataFrame(result_rows))


def compute_land_use_emissions(input_data):
    """Compute land use (AFOLU) emissions.

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Input data with AFOLU emissions.

    Returns
    -------
    pyam.IamDataFrame
        Land use emissions in Mt CO2/yr.
        Variable name: "Emissions|CO2|Land Use"
    """
    data = input_data.data
    result_rows = []

    # Get AFOLU emissions
    afolu_data = data[data["variable"] == input_variables.EMISSIONS_CO2_AFOLU]

    for _, row in afolu_data.iterrows():
        result_rows.append({
            "model": row["model"],
            "scenario": row["scenario"],
            "region": row["region"],
            "variable": "Emissions|CO2|Land Use",
            "unit": "Mt CO2/yr",
            "year": row["year"],
            "value": row["value"],
        })

    return pyam.IamDataFrame(pd.DataFrame(result_rows))


def compute_all_sectors_emissions(input_data):
    """Compute total emissions breakdown for all sectors.

    Computes:
    - NFC (Net Fossil Carbon) - from Kaya decomposition
    - NIC (Net Industrial Carbon) - industrial processes
    - Other Gases (CH4 + N2O + F-gases in CO2-eq)
    - Land Use (AFOLU)
    - Total Net Emissions

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Input data with all required variables.

    Returns
    -------
    pyam.IamDataFrame
        All emission components.
    """
    # Compute Kaya variables to get NFC
    kaya_vars = compute_kaya_variables(input_data)

    # Extract NFC
    nfc = kaya_vars.filter(variable=kaya_variable_names.NFC)

    # Compute other components
    nic = compute_industrial_process_emissions(input_data)
    other_gases = compute_other_gases_emissions(input_data)
    land_use = compute_land_use_emissions(input_data)

    # Combine all
    result = pyam.concat([nfc, nic, other_gases, land_use])

    return result


def _compute_sector_lmdi_cumulative(sector_data, base_year, variable_name, output_name):
    """Compute cumulative LMDI-style difference for a non-Kaya sector.

    For non-Kaya sectors, we just compute the difference from base year
    (not a true LMDI decomposition, since there's no underlying identity).

    Parameters
    ----------
    sector_data : pyam.IamDataFrame
        Sector emissions data.
    base_year : int
        Base year for comparison.
    variable_name : str
        Variable name to extract from sector_data.
    output_name : str
        Output variable name for the LMDI-style contribution.

    Returns
    -------
    pyam.IamDataFrame
        Sector contribution at each year (difference from base year).
    """
    data = sector_data.data
    var_data = data[data["variable"] == variable_name]

    if len(var_data) == 0:
        return None

    # Get base year value
    base_data = var_data[var_data["year"] == base_year]
    if len(base_data) == 0:
        return None

    base_value = base_data["value"].values[0]

    result_rows = []
    for _, row in var_data.iterrows():
        diff = row["value"] - base_value
        result_rows.append({
            "model": row["model"],
            "scenario": row["scenario"],
            "region": row["region"],
            "variable": output_name,
            "unit": "Mt CO2/yr",
            "year": row["year"],
            "value": diff,
        })

    return pyam.IamDataFrame(pd.DataFrame(result_rows))


# Use shared trapezoidal integration function
_trapezoidal_integrate_sector = trapezoidal_integrate


def _compute_sector_period_sum(sector_lmdi, periods, integration_method="trapezoidal"):
    """Compute period sums for a non-Kaya sector.

    Parameters
    ----------
    sector_lmdi : pyam.IamDataFrame
        Output from _compute_sector_lmdi_cumulative.
    periods : list of tuples
        List of (start_year, end_year) periods.
    integration_method : str
        "trapezoidal" or "endpoint".

    Returns
    -------
    dict
        Dictionary mapping period labels to sum values in Gt CO2.
    """
    if sector_lmdi is None:
        return None

    data = sector_lmdi.data
    variable = data["variable"].iloc[0]

    result = {}
    for start_year, end_year in periods:
        period_label = f"{start_year} to {end_year}"

        if integration_method == "trapezoidal":
            var_data = data.sort_values("year")
            period_sum = _trapezoidal_integrate_sector(var_data, start_year, end_year)
        else:
            available_years = sorted(data["year"].unique())
            years_in_period = [y for y in available_years if start_year < y <= end_year]
            period_sum = data[data["year"].isin(years_in_period)]["value"].sum()

        # Convert Mt to Gt
        result[period_label] = period_sum / 1000

    return {variable: result}


def compute_all_sectors_lmdi_cumulative(
    input_data,
    base_year=2020,
    scenario=None,
    periods=None,
    integration_method="trapezoidal",
    use_corrected=False,
):
    """Compute cumulative LMDI for all emission sectors.

    This is the main entry point for reproducing the "Fig Ref drivers"
    analysis from the Excel file. Returns a table matching the
    LMDItableRefAllSectors format.

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Raw input data with all required variables.
    base_year : int
        Base year for LMDI calculation (default 2020).
    scenario : tuple (model, scenario, region), optional
        Scenario to analyze. If None, uses first available.
    periods : list of tuples, optional
        Periods to sum over. Default: [(2020, 2050), (2050, 2100), (2020, 2100)]
    integration_method : str, optional
        Method for computing period sums:
        - "trapezoidal" (default): Trapezoidal integration, matches Excel methodology
        - "endpoint": Simple sum of endpoint values (legacy behavior)
    use_corrected : bool, optional
        If True, use corrected (non-negative) values. Default False matches Excel.

    Returns
    -------
    pd.DataFrame
        Table with rows:
        - Population
        - Economic Activity per Person
        - Energy Intensity of Economy
        - Energy Supply Loss Factor
        - Fossil Fuel Fraction
        - Carbon Intensity of Fossil Energy
        - Industrial Process Carbon Emissions
        - Other Gases
        - Land Use
        - Total Net Emissions

        Columns are period labels (e.g., "2020 to 2050").
        Values are in Gt CO2.

    Example
    -------
    >>> import pyam
    >>> data = pyam.IamDataFrame("scenario_data.csv")
    >>> result = compute_all_sectors_lmdi_cumulative(
    ...     data,
    ...     base_year=2020,
    ...     scenario=("IMAGE 3.0.1", "SSP2-Baseline", "World"),
    ... )
    >>> print(result)
    """
    if periods is None:
        periods = [(2020, 2050), (2050, 2100), (2020, 2100)]

    # Filter to scenario if specified
    if scenario is not None:
        input_data = input_data.filter(
            model=scenario[0], scenario=scenario[1], region=scenario[2]
        )
    else:
        # Use first available scenario with bounds check
        data = input_data.data
        if data.empty:
            raise ValueError("Input data is empty. Cannot compute LMDI analysis.")

        first_model = data["model"].iloc[0]
        first_scenario = data["scenario"].iloc[0]
        first_region = data["region"].iloc[0]
        input_data = input_data.filter(
            model=first_model, scenario=first_scenario, region=first_region
        )
        scenario = (first_model, first_scenario, first_region)

    # 1. Compute Kaya variables and factors
    kaya_vars = compute_kaya_variables(input_data)
    kaya_factors = compute_kaya_factors(kaya_vars)

    # 2. Compute cumulative LMDI sum for Kaya factors using new method
    result = compute_lmdi_cumulative_sum(
        kaya_factors,
        base_year=base_year,
        periods=periods,
        integration_method=integration_method,
        use_corrected=use_corrected,
    )

    # 3. Compute contributions for additional sectors with trapezoidal integration
    # Industrial Process Carbon
    nic = compute_industrial_process_emissions(input_data)
    nic_lmdi = _compute_sector_lmdi_cumulative(
        nic, base_year, "Net Industrial Carbon",
        lmdi_cumulative_names.Industrial_Process
    )
    nic_sums = _compute_sector_period_sum(nic_lmdi, periods, integration_method)

    # Other Gases
    other_gases = compute_other_gases_emissions(input_data)
    other_gases_lmdi = _compute_sector_lmdi_cumulative(
        other_gases, base_year, "Emissions|Other Gases|CO2-equivalent",
        lmdi_cumulative_names.Other_Gases
    )
    other_gases_sums = _compute_sector_period_sum(other_gases_lmdi, periods, integration_method)

    # Land Use
    land_use = compute_land_use_emissions(input_data)
    land_use_lmdi = _compute_sector_lmdi_cumulative(
        land_use, base_year, "Emissions|CO2|Land Use",
        lmdi_cumulative_names.Land_Use
    )
    land_use_sums = _compute_sector_period_sum(land_use_lmdi, periods, integration_method)

    # 4. Add non-Kaya sector rows to result
    for sector_sums in [nic_sums, other_gases_sums, land_use_sums]:
        if sector_sums is not None:
            for var_name, period_values in sector_sums.items():
                for period_label, value in period_values.items():
                    result.loc[var_name, period_label] = value

    # 5. Add Total Net Emissions row
    total_row = result.sum(axis=0)
    result.loc[lmdi_cumulative_names.Total_Net_Emissions] = total_row

    # Reorder rows to match expected format
    standard_order = [
        lmdi_cumulative_names.Pop_cumulative,
        lmdi_cumulative_names.GNP_per_P_cumulative,
        lmdi_cumulative_names.FE_per_GNP_cumulative,
        lmdi_cumulative_names.PEdeq_per_FE_cumulative,
        lmdi_cumulative_names.PEFF_per_PEDEq_cumulative,
        lmdi_cumulative_names.TFC_per_PEFF_cumulative,
        lmdi_cumulative_names.Industrial_Process,
        lmdi_cumulative_names.Other_Gases,
        lmdi_cumulative_names.Land_Use,
        lmdi_cumulative_names.Total_Net_Emissions,
    ]

    existing_order = [v for v in standard_order if v in result.index]
    result = result.reindex(existing_order)

    return result

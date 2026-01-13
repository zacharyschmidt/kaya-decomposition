"""Compute savings (avoided emissions) from comparing two scenarios.

This module implements the "Savings" tab analysis that compares a Reference
scenario to an Intervention scenario to quantify avoided emissions and
attribute them to different factors.
"""

import warnings

import numpy as np
import pandas as pd

import pyam
from kaya_decomposition.constants import (
    input_variables,
    kaya_factors as kaya_factor_names,
    kaya_variables as kaya_variable_names,
    savings as savings_names,
    lmdi_cumulative as lmdi_cumulative_names,
)
from kaya_decomposition.variables import compute_kaya_variables
from kaya_decomposition.factors import compute_kaya_factors
from kaya_decomposition.all_sectors import (
    compute_other_gases_emissions,
    compute_industrial_process_emissions,
    compute_total_industrial_carbon,
    compute_land_use_emissions,
)
from kaya_decomposition.utils import trapezoidal_integrate


# Mapping from Kaya factors to savings output names
FACTOR_TO_SAVINGS_NAME = {
    input_variables.POPULATION: savings_names.POPULATION,
    kaya_factor_names.GNP_per_P: savings_names.ECONOMIC_ACTIVITY,
    kaya_factor_names.FE_per_GNP: savings_names.ENERGY_INTENSITY,
    kaya_factor_names.PEdeq_per_FE: savings_names.ENERGY_SUPPLY_LOSS,
    kaya_factor_names.PEFF_per_PEDEq: savings_names.FOSSIL_FRACTION,
    kaya_factor_names.TFC_per_PEFF: savings_names.CARBON_INTENSITY,
}


def _logarithmic_mean(a, b):
    """Calculate the logarithmic mean of two values.

    L(a, b) = (a - b) / (ln(a) - ln(b))

    Special cases:
    - If a == b: returns a (by L'Hôpital's rule)
    - If a <= 0 or b <= 0: returns NaN
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        result = np.where(
            np.isclose(a, b),
            a,
            (a - b) / (np.log(a) - np.log(b))
        )

        result = np.where((a <= 0) | (b <= 0), np.nan, result)

    return result


def compute_lmdi_scenario_comparison(
    kaya_factors_df,
    ref_scenario,
    int_scenario,
):
    """Compute LMDI decomposition between two scenarios at each time point.

    Unlike compute_lmdi() which returns LMDI for matched time points,
    this computes the decomposition for ALL time points in the data,
    enabling subsequent integration over time.

    Parameters
    ----------
    kaya_factors_df : pyam.IamDataFrame
        Output from compute_kaya_factors() for both scenarios.
    ref_scenario : tuple (model, scenario, region)
        Reference scenario identifiers.
    int_scenario : tuple (model, scenario, region)
        Intervention scenario identifiers.

    Returns
    -------
    pyam.IamDataFrame
        LMDI contributions at each time point.
        Scenario name is "{ref}::{int}".
    """
    # Extract reference and intervention data
    ref_data = kaya_factors_df.filter(
        model=ref_scenario[0], scenario=ref_scenario[1], region=ref_scenario[2]
    ).data
    int_data = kaya_factors_df.filter(
        model=int_scenario[0], scenario=int_scenario[1], region=int_scenario[2]
    ).data

    # Get all years present in both scenarios
    ref_years = set(ref_data["year"].unique())
    int_years = set(int_data["year"].unique())
    years = sorted(ref_years & int_years)

    # Combined scenario name
    combined_model = f"{ref_scenario[0]}::{int_scenario[0]}"
    combined_scenario = f"{ref_scenario[1]}::{int_scenario[1]}"
    combined_region = f"{ref_scenario[2]}::{int_scenario[2]}"

    result_rows = []

    for year in years:
        # Get TFC values for this year
        tfc_ref = ref_data[
            (ref_data["variable"] == kaya_variable_names.TFC) &
            (ref_data["year"] == year)
        ]["value"].values
        tfc_int = int_data[
            (int_data["variable"] == kaya_variable_names.TFC) &
            (int_data["year"] == year)
        ]["value"].values

        if len(tfc_ref) == 0 or len(tfc_int) == 0:
            continue

        tfc_ref = tfc_ref[0]
        tfc_int = tfc_int[0]

        # Calculate logarithmic mean
        log_mean = _logarithmic_mean(tfc_ref, tfc_int)

        # Calculate LMDI term for each factor
        for factor_name, output_name in FACTOR_TO_SAVINGS_NAME.items():
            # Get factor values
            factor_ref = ref_data[
                (ref_data["variable"] == factor_name) & (ref_data["year"] == year)
            ]["value"].values
            factor_int = int_data[
                (int_data["variable"] == factor_name) & (int_data["year"] == year)
            ]["value"].values

            if len(factor_ref) == 0 or len(factor_int) == 0:
                continue

            factor_ref = factor_ref[0]
            factor_int = factor_int[0]

            # LMDI contribution: L(TFC_ref, TFC_int) * ln(factor_ref / factor_int)
            # Note: This gives POSITIVE values when ref > int (i.e., savings)
            if factor_ref > 0 and factor_int > 0:
                lmdi_value = log_mean * np.log(factor_ref / factor_int)
            else:
                lmdi_value = np.nan

            result_rows.append({
                "model": combined_model,
                "scenario": combined_scenario,
                "region": combined_region,
                "variable": output_name,
                "unit": "Mt CO2/yr",
                "year": year,
                "value": lmdi_value,
            })

    return pyam.IamDataFrame(pd.DataFrame(result_rows))


def compute_cumulative_emissions(
    data,
    variable_name,
    start_year,
    end_year,
    integration_method="trapezoidal",
):
    """Compute cumulative emissions for a single variable over a period.

    Parameters
    ----------
    data : pd.DataFrame
        Data with 'year', 'value' columns.
    variable_name : str
        Variable to integrate.
    start_year : int
        Start of period.
    end_year : int
        End of period.
    integration_method : str
        "trapezoidal" or "endpoint".

    Returns
    -------
    float
        Cumulative emissions in original units (e.g., Mt CO2).
    """
    var_data = data[data["variable"] == variable_name].sort_values("year")

    if len(var_data) == 0:
        return 0.0

    if integration_method == "trapezoidal":
        return trapezoidal_integrate(var_data, start_year, end_year)
    else:
        available_years = sorted(var_data["year"].unique())
        years_in_period = [y for y in available_years if start_year <= y <= end_year]
        return var_data[var_data["year"].isin(years_in_period)]["value"].sum()


def compute_total_anthropogenic_emissions(
    input_data,
    scenario,
    start_year,
    end_year,
    integration_method="trapezoidal",
):
    """Compute cumulative total anthropogenic emissions for a scenario.

    Total = TFC + Industrial Process + Other Gases + Land Use

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Raw input data.
    scenario : tuple (model, scenario, region)
        Scenario identifiers.
    start_year : int
        Start of period.
    end_year : int
        End of period.
    integration_method : str
        "trapezoidal" or "endpoint".

    Returns
    -------
    float
        Cumulative emissions in Gt CO2-equivalent.
    """
    # Filter to scenario
    filtered = input_data.filter(
        model=scenario[0], scenario=scenario[1], region=scenario[2]
    )

    # Compute Kaya variables to get TFC
    kaya_vars = compute_kaya_variables(filtered)
    tfc_data = kaya_vars.filter(variable=kaya_variable_names.TFC).data

    # Get Industrial Process emissions
    ip = compute_industrial_process_emissions(filtered)
    ip_data = ip.data

    # Get Other Gases
    og = compute_other_gases_emissions(filtered)
    og_data = og.data

    # Get Land Use
    lu = compute_land_use_emissions(filtered)
    lu_data = lu.data

    # Integrate each component
    tfc_cum = compute_cumulative_emissions(
        tfc_data, kaya_variable_names.TFC, start_year, end_year, integration_method
    )
    ip_cum = compute_cumulative_emissions(
        ip_data, "Net Industrial Carbon", start_year, end_year, integration_method
    )
    og_cum = compute_cumulative_emissions(
        og_data, "Emissions|Other Gases|CO2-equivalent", start_year, end_year, integration_method
    )
    lu_cum = compute_cumulative_emissions(
        lu_data, "Emissions|CO2|Land Use", start_year, end_year, integration_method
    )

    # Convert Mt to Gt and return total
    total = (tfc_cum + ip_cum + og_cum + lu_cum) / 1000
    return total


def _compute_sector_difference(
    ref_data,
    int_data,
    variable_name,
    output_name,
):
    """Compute difference between scenarios for a non-Kaya sector.

    For sectors like Industry, Land Use, and Other Gases, we compute:
    contribution(t) = sector_ref(t) - sector_int(t)

    This gives POSITIVE values when ref > int (i.e., when intervention reduces emissions).

    Parameters
    ----------
    ref_data : pyam.IamDataFrame
        Reference scenario sector data.
    int_data : pyam.IamDataFrame
        Intervention scenario sector data.
    variable_name : str
        Variable to extract.
    output_name : str
        Output variable name.

    Returns
    -------
    pyam.IamDataFrame
        Sector difference at each year.
    """
    ref = ref_data.filter(variable=variable_name).data
    int_ = int_data.filter(variable=variable_name).data

    if len(ref) == 0 or len(int_) == 0:
        return None

    # Get years in both
    ref_years = set(ref["year"].unique())
    int_years = set(int_["year"].unique())
    years = sorted(ref_years & int_years)

    result_rows = []
    for year in years:
        ref_val = ref[ref["year"] == year]["value"].values[0]
        int_val = int_[int_["year"] == year]["value"].values[0]
        diff = ref_val - int_val

        result_rows.append({
            "model": "LMDI",
            "scenario": "LMDI",
            "region": "World",
            "variable": output_name,
            "unit": "Mt CO2/yr",
            "year": year,
            "value": diff,
        })

    return pyam.IamDataFrame(pd.DataFrame(result_rows))


def compute_intervention_ccs(
    input_data,
    int_scenario,
    start_year,
    end_year,
    integration_method="trapezoidal",
):
    """Compute cumulative CCS contributions from Intervention scenario.

    CCS represents additional emissions reduction beyond what's captured
    in the Kaya decomposition. We separate into:
    - Fossil CCS (Energy + Industry)
    - Biomass CCS (Energy + Industry)

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Raw input data containing Intervention scenario.
    int_scenario : tuple (model, scenario, region)
        Intervention scenario identifiers.
    start_year : int
        Start of period.
    end_year : int
        End of period.
    integration_method : str
        "trapezoidal" or "endpoint".

    Returns
    -------
    dict
        {"fossil_ccs": float, "biomass_ccs": float} in Gt CO2.
    """
    # Filter to intervention scenario
    filtered = input_data.filter(
        model=int_scenario[0], scenario=int_scenario[1], region=int_scenario[2]
    )
    data = filtered.data

    # Fossil CCS = Energy + Industry
    fossil_energy_ccs = compute_cumulative_emissions(
        data, input_variables.CCS_FOSSIL_ENERGY, start_year, end_year, integration_method
    )
    fossil_industry_ccs = compute_cumulative_emissions(
        data, input_variables.CCS_FOSSIL_INDUSTRY, start_year, end_year, integration_method
    )

    # Biomass CCS = Energy + Industry
    biomass_energy_ccs = compute_cumulative_emissions(
        data, input_variables.CCS_BIOMASS_ENERGY, start_year, end_year, integration_method
    )
    biomass_industry_ccs = compute_cumulative_emissions(
        data, input_variables.CCS_BIOMASS_INDUSTRY, start_year, end_year, integration_method
    )

    # Convert Mt to Gt (note: CCS is reported as positive sequestration, so negate)
    fossil_ccs = -(fossil_energy_ccs + fossil_industry_ccs) / 1000
    biomass_ccs = -(biomass_energy_ccs + biomass_industry_ccs) / 1000

    return {"fossil_ccs": fossil_ccs, "biomass_ccs": biomass_ccs}


def compute_savings(
    input_data,
    ref_scenario,
    int_scenario,
    periods=None,
    integration_method="trapezoidal",
):
    """Compute avoided emissions (savings) from comparing two scenarios.

    This is the main entry point for computing the "Savings" table
    that matches the Excel Savings tab format.

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Raw input data containing both Reference and Intervention scenarios.
    ref_scenario : tuple (model, scenario, region)
        Reference scenario identifiers.
    int_scenario : tuple (model, scenario, region)
        Intervention scenario identifiers.
    periods : list of tuples, optional
        Periods to compute savings for.
        Default: [(2020, 2050), (2050, 2100), (2020, 2100)]
    integration_method : str
        "trapezoidal" (default, matches Excel) or "endpoint".

    Returns
    -------
    pd.DataFrame
        Table with rows for each emission component and columns for:
        - Absolute values (Gt CO2) for each period
        - Percentages (if single period requested)

        Rows include:
        - Reference case cumulative emissions
        - Intervention case cumulative emissions
        - Difference
        - Kaya factor contributions (Population, Economic Activity, etc.)
        - Non-Kaya sectors (Industry, Land Use, Other Gases)
        - CCS contributions (Fossil CCS, Biomass CCS)
        - Total/Net
    """
    if periods is None:
        periods = [(2020, 2050), (2050, 2100), (2020, 2100)]

    # Compute Kaya variables and factors for both scenarios
    kaya_vars = compute_kaya_variables(input_data)
    kaya_factors = compute_kaya_factors(kaya_vars)

    # Compute LMDI comparison between scenarios
    lmdi_comparison = compute_lmdi_scenario_comparison(
        kaya_factors, ref_scenario, int_scenario
    )

    # Compute sector data for both scenarios
    ref_filtered = input_data.filter(
        model=ref_scenario[0], scenario=ref_scenario[1], region=ref_scenario[2]
    )
    int_filtered = input_data.filter(
        model=int_scenario[0], scenario=int_scenario[1], region=int_scenario[2]
    )

    # Industrial Process (using Total Industrial Carbon for consistency with Kaya)
    ref_tic = compute_total_industrial_carbon(ref_filtered)
    int_tic = compute_total_industrial_carbon(int_filtered)
    tic_diff = _compute_sector_difference(
        ref_tic, int_tic, "Total Industrial Carbon", savings_names.INDUSTRIAL_PROCESS
    )

    # Other Gases
    ref_og = compute_other_gases_emissions(ref_filtered)
    int_og = compute_other_gases_emissions(int_filtered)
    og_diff = _compute_sector_difference(
        ref_og, int_og, "Emissions|Other Gases|CO2-equivalent", savings_names.OTHER_GASES
    )

    # Land Use
    ref_lu = compute_land_use_emissions(ref_filtered)
    int_lu = compute_land_use_emissions(int_filtered)
    lu_diff = _compute_sector_difference(
        ref_lu, int_lu, "Emissions|CO2|Land Use", savings_names.LAND_USE
    )

    # Build result DataFrame
    result_data = {}

    for start_year, end_year in periods:
        period_label = f"{start_year} to {end_year}"
        period_values = {}

        # Compute cumulative emissions for both scenarios
        ref_cumulative = compute_total_anthropogenic_emissions(
            input_data, ref_scenario, start_year, end_year, integration_method
        )
        int_cumulative = compute_total_anthropogenic_emissions(
            input_data, int_scenario, start_year, end_year, integration_method
        )
        difference = ref_cumulative - int_cumulative

        period_values[savings_names.REF_CUMULATIVE] = ref_cumulative
        period_values[savings_names.INT_CUMULATIVE] = int_cumulative
        period_values[savings_names.DIFFERENCE] = difference

        # Integrate LMDI factor contributions
        lmdi_data = lmdi_comparison.data
        for output_name in FACTOR_TO_SAVINGS_NAME.values():
            var_data = lmdi_data[lmdi_data["variable"] == output_name].sort_values("year")
            if len(var_data) > 0:
                if integration_method == "trapezoidal":
                    contrib = trapezoidal_integrate(var_data, start_year, end_year) / 1000
                else:
                    years_in_period = var_data[
                        (var_data["year"] >= start_year) & (var_data["year"] <= end_year)
                    ]
                    contrib = years_in_period["value"].sum() / 1000
                period_values[output_name] = contrib
            else:
                period_values[output_name] = 0.0

        # Integrate sector contributions
        for sector_diff, sector_name in [
            (tic_diff, savings_names.INDUSTRIAL_PROCESS),
            (og_diff, savings_names.OTHER_GASES),
            (lu_diff, savings_names.LAND_USE),
        ]:
            if sector_diff is not None:
                var_data = sector_diff.data.sort_values("year")
                if integration_method == "trapezoidal":
                    contrib = trapezoidal_integrate(var_data, start_year, end_year) / 1000
                else:
                    years_in_period = var_data[
                        (var_data["year"] >= start_year) & (var_data["year"] <= end_year)
                    ]
                    contrib = years_in_period["value"].sum() / 1000
                period_values[sector_name] = contrib
            else:
                period_values[sector_name] = 0.0

        # Compute CCS contributions from Intervention
        ccs = compute_intervention_ccs(
            input_data, int_scenario, start_year, end_year, integration_method
        )
        period_values[savings_names.FOSSIL_CCS] = ccs["fossil_ccs"]
        period_values[savings_names.BIOMASS_CCS] = ccs["biomass_ccs"]

        # Calculate Total/Net as sum of all contributions
        total = sum([
            period_values.get(savings_names.POPULATION, 0),
            period_values.get(savings_names.ECONOMIC_ACTIVITY, 0),
            period_values.get(savings_names.ENERGY_INTENSITY, 0),
            period_values.get(savings_names.ENERGY_SUPPLY_LOSS, 0),
            period_values.get(savings_names.FOSSIL_FRACTION, 0),
            period_values.get(savings_names.CARBON_INTENSITY, 0),
            period_values.get(savings_names.INDUSTRIAL_PROCESS, 0),
            period_values.get(savings_names.OTHER_GASES, 0),
            period_values.get(savings_names.LAND_USE, 0),
            period_values.get(savings_names.FOSSIL_CCS, 0),
            period_values.get(savings_names.BIOMASS_CCS, 0),
        ])
        period_values[savings_names.TOTAL_NET] = total

        result_data[period_label] = period_values

    # Create DataFrame
    result_df = pd.DataFrame(result_data)

    # Order rows
    row_order = [
        savings_names.REF_CUMULATIVE,
        savings_names.INT_CUMULATIVE,
        savings_names.DIFFERENCE,
        savings_names.ENERGY_SUPPLY_LOSS,
        savings_names.POPULATION,
        savings_names.ECONOMIC_ACTIVITY,
        savings_names.ENERGY_INTENSITY,
        savings_names.FOSSIL_FRACTION,
        savings_names.CARBON_INTENSITY,
        savings_names.INDUSTRIAL_PROCESS,
        savings_names.LAND_USE,
        savings_names.OTHER_GASES,
        savings_names.FOSSIL_CCS,
        savings_names.BIOMASS_CCS,
        savings_names.TOTAL_NET,
    ]
    existing_order = [r for r in row_order if r in result_df.index]
    result_df = result_df.reindex(existing_order)

    return result_df


def compute_savings_with_percentages(
    input_data,
    ref_scenario,
    int_scenario,
    period=(2020, 2100),
    integration_method="trapezoidal",
):
    """Compute savings with percentage columns matching Excel format.

    This is a convenience function that returns a single-period savings
    table with additional percentage columns.

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Raw input data containing both Reference and Intervention scenarios.
    ref_scenario : tuple (model, scenario, region)
        Reference scenario identifiers.
    int_scenario : tuple (model, scenario, region)
        Intervention scenario identifiers.
    period : tuple (start_year, end_year)
        Period to compute savings for.
    integration_method : str
        "trapezoidal" (default) or "endpoint".

    Returns
    -------
    pd.DataFrame
        Table with columns:
        - "Gt CO2": Absolute values
        - "% of total savings": Percentage of total avoided emissions
        - "% of reference emissions": Percentage of reference cumulative emissions
    """
    # Get basic savings table
    savings_df = compute_savings(
        input_data, ref_scenario, int_scenario,
        periods=[period], integration_method=integration_method
    )

    period_label = f"{period[0]} to {period[1]}"
    abs_values = savings_df[period_label]

    # Calculate percentages
    ref_cumulative = abs_values[savings_names.REF_CUMULATIVE]
    difference = abs_values[savings_names.DIFFERENCE]

    # Contribution rows (exclude summary rows)
    contribution_rows = [
        savings_names.ENERGY_SUPPLY_LOSS,
        savings_names.POPULATION,
        savings_names.ECONOMIC_ACTIVITY,
        savings_names.ENERGY_INTENSITY,
        savings_names.FOSSIL_FRACTION,
        savings_names.CARBON_INTENSITY,
        savings_names.INDUSTRIAL_PROCESS,
        savings_names.LAND_USE,
        savings_names.OTHER_GASES,
        savings_names.FOSSIL_CCS,
        savings_names.BIOMASS_CCS,
        savings_names.TOTAL_NET,
    ]

    # Build result with percentage columns
    result = pd.DataFrame()
    result[savings_names.ABS_VALUE] = abs_values

    # % of total savings
    pct_of_total = pd.Series(index=abs_values.index, dtype=float)
    for row in contribution_rows:
        if row in abs_values.index and difference != 0:
            # Negate because we want positive % for negative contributions
            pct_of_total[row] = -abs_values[row] / difference * 100
    result[savings_names.PCT_OF_TOTAL] = pct_of_total

    # % of reference emissions
    pct_of_ref = pd.Series(index=abs_values.index, dtype=float)
    for row in contribution_rows:
        if row in abs_values.index and ref_cumulative != 0:
            pct_of_ref[row] = abs_values[row] / ref_cumulative * 100
    result[savings_names.PCT_OF_REF] = pct_of_ref

    return result

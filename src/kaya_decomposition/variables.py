"""Compute intermediate Kaya variables."""

import logging
import warnings

import pyam
from kaya_decomposition.constants import input_variables, kaya_variables

logger = logging.getLogger(__name__)

# Use the explicit list of required variables from input_variables module
required_input_variables = input_variables.REQUIRED_VARIABLES


def compute_kaya_variables(input_data):
    """Compute Kaya intermediate variables from input data.

    Parameters
    ----------
    input_data : pyam.IamDataFrame
        Input data containing required variables.

    Returns
    -------
    pyam.IamDataFrame
        IamDataFrame with computed Kaya variables.

    Raises
    ------
    ValueError
        If required input variables are missing from the input data.

    Notes
    -----
    Required input variables:

    - Population
    - GDP (MER or PPP)
    - Final Energy
    - Primary Energy
    - Primary Energy|Coal, Oil, Gas
    - Various CO2 emissions and CCS variables

    Computed variables:

    - Primary Energy|Fossil
    - Total Fossil Carbon
    - Net Fossil Carbon
    """
    missing = _get_missing_variables(input_data)
    if missing:
        raise ValueError(
            f"Cannot compute Kaya variables: missing required input variables. "
            f"Missing: {sorted(missing)}"
        )

    kaya_vars = pyam.concat(
        [
            _calc_pop(input_data),
            _calc_gdp(input_data),
            _calc_fe(input_data),
            _calc_pe(input_data),
            _calc_pe_ff(input_data),
            _calc_tfc(input_data),
            _calc_nfc(input_data),
        ]
    )
    return kaya_vars


def _get_missing_variables(input_data):
    """Return set of missing required variables, or empty set if complete."""
    input_vars = set(input_data.data["variable"].unique())
    required_variables_set = make_required_variables_set(input_vars)
    missing = required_variables_set - input_vars

    # Log missing variables per model/scenario/region for debugging
    if missing:
        scenario_model_region = input_data.data[
            ["model", "scenario", "region"]
        ].drop_duplicates()
        for _, row in scenario_model_region.iterrows():
            single_combination = input_data.filter(
                model=row["model"], scenario=row["scenario"], region=row["region"]
            )
            single_vars = set(single_combination.data["variable"].unique())
            single_required = make_required_variables_set(single_vars)
            single_missing = single_required - single_vars
            if single_missing:
                logger.info(
                    f"""Variables missing for
                model: {row['model']},
                scenario: {row['scenario']},
                region: {row['region']}\nMissing variables: {single_missing}"""
                )

    return missing


def _is_input_data_incomplete(input_data):
    """Check if input data is missing required variables."""
    return bool(_get_missing_variables(input_data))


def make_required_variables_set(input_vars):
    required_variables_set = set(required_input_variables)
    if _has_at_least_one_gdp(input_vars):
        # either form of GDP is acceptable, so don't check for both
        # as long as one is present
        return required_variables_set - set(
            [input_variables.GDP_PPP, input_variables.GDP_MER]
        )
    return required_variables_set


def _has_at_least_one_gdp(input_vars):
    return (
        input_variables.GDP_PPP in input_vars
        or input_variables.GDP_MER in input_vars
    )


def _calc_pop(input_data):
    return input_data.filter(variable=input_variables.POPULATION)


def _calc_gdp(input_data):
    variable = input_variables.GDP_PPP
    if input_data.filter(variable=variable).empty:
        variable = input_variables.GDP_MER
    return input_data.filter(variable=variable)


def _calc_fe(input_data):
    return input_data.filter(variable=input_variables.FINAL_ENERGY)


def _calc_pe(input_data):
    return input_data.filter(variable=input_variables.PRIMARY_ENERGY)


def _calc_pe_ff(input_data):
    input_data = input_data.copy()
    input_data.add(
        input_variables.PRIMARY_ENERGY_COAL,
        input_variables.PRIMARY_ENERGY_OIL,
        "pe_coal_oil",
        append=True,
    )
    return input_data.add(
        input_variables.PRIMARY_ENERGY_GAS,
        "pe_coal_oil",
        kaya_variables.PRIMARY_ENERGY_FF,
    )


def _calc_nfc(input_data):
    input_data = input_data.copy()
    input_data.subtract(
        input_variables.EMISSIONS_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES,
        input_variables.EMISSIONS_CO2_INDUSTRIAL_PROCESSES,
        "net_energy_emissions_with_biomass_ccs",
        ignore_units="Mt CO2/yr",
        append=True,
    )
    return input_data.add(
        input_variables.CCS_BIOMASS,
        "net_energy_emissions_with_biomass_ccs",
        kaya_variables.NFC,
        ignore_units="Mt CO2/yr",
        append=False,
    )


def _calc_tfc(input_data):
    input_data = input_data.copy()
    ccs_fossil_energy = _calc_ccs_fossil_energy(input_data)
    nfc = _calc_nfc(input_data)
    nfc_with_ccs_fossil_energy = nfc.append(ccs_fossil_energy)
    return nfc_with_ccs_fossil_energy.add(
        "ccs_fossil_energy",
        kaya_variables.NFC,
        kaya_variables.TFC,
        ignore_units="Mt CO2/yr",
    )


def _calc_ccs_fossil_energy(input_data):
    input_data = input_data.copy()
    input_data.subtract(
        input_variables.CCS,
        input_variables.CCS_BIOMASS,
        "ccs_fossil",
        ignore_units="Mt CO2/yr",
        append=True,
    )
    return input_data.subtract(
        "ccs_fossil",
        input_variables.CCS_FOSSIL_INDUSTRY,
        "ccs_fossil_energy",
        ignore_units="Mt CO2/yr",
    )

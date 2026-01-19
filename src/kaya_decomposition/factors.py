"""Compute Kaya decomposition factors."""

import pyam
from kaya_decomposition.constants import input_variables, kaya_factors, kaya_variables


def compute_kaya_factors(kaya_variables_frame):
    """Compute Kaya decomposition factors.

    Parameters
    ----------
    kaya_variables_frame : pyam.IamDataFrame
        IamDataFrame containing Kaya intermediate variables
        (output from compute_kaya_variables).

    Returns
    -------
    pyam.IamDataFrame
        IamDataFrame with computed Kaya factors.

    Raises
    ------
    ValueError
        If kaya_variables_frame is None, not an IamDataFrame, or empty.

    Notes
    -----
    Computed factors:

    - GNP/P (GDP per capita)
    - FE/GNP (Energy intensity of GDP)
    - PEDEq/FE (Primary energy to final energy ratio)
    - PEFF/PEDEq (Fossil share of primary energy)
    - TFC/PEFF (Carbon intensity of fossil energy)
    - NFC/TFC (Net to total fossil carbon ratio)
    """
    if kaya_variables_frame is None:
        raise ValueError(
            "Cannot compute Kaya factors: kaya_variables_frame is None. "
            "Ensure compute_kaya_variables() returned valid data."
        )
    if not isinstance(kaya_variables_frame, pyam.IamDataFrame):
        raise ValueError(
            f"Cannot compute Kaya factors: expected pyam.IamDataFrame, "
            f"got {type(kaya_variables_frame).__name__}"
        )
    if kaya_variables_frame.empty:
        raise ValueError("Cannot compute Kaya factors: input IamDataFrame is empty.")

    factors = pyam.concat(
        [
            _calc_gnp_per_p(kaya_variables_frame),
            _calc_fe_per_gnp(kaya_variables_frame),
            _calc_pedeq_per_fe(kaya_variables_frame),
            _calc_peff_per_pedeq(kaya_variables_frame),
            _calc_tfc_per_peff(kaya_variables_frame),
            _calc_nfc_per_tfc(kaya_variables_frame),
            kaya_variables_frame.filter(
                variable=[kaya_variables.TFC, input_variables.POPULATION]
            ),
        ]
    )
    return factors


def _calc_gnp_per_p(input_data):
    variable = input_variables.GDP_PPP
    if input_data.filter(variable=variable).empty:
        variable = input_variables.GDP_MER
    return input_data.divide(
        variable,
        input_variables.POPULATION,
        kaya_factors.GNP_per_P,
        append=False,
    )


def _calc_fe_per_gnp(input_data):
    variable = input_variables.GDP_PPP
    if input_data.filter(variable=variable).empty:
        variable = input_variables.GDP_MER
    return input_data.divide(
        input_variables.FINAL_ENERGY,
        variable,
        kaya_factors.FE_per_GNP,
        append=False,
    )


def _calc_pedeq_per_fe(input_data):
    return input_data.divide(
        input_variables.PRIMARY_ENERGY,
        input_variables.FINAL_ENERGY,
        kaya_factors.PEdeq_per_FE,
        append=False,
    )


def _calc_peff_per_pedeq(input_data):
    return input_data.divide(
        kaya_variables.PRIMARY_ENERGY_FF,
        input_variables.PRIMARY_ENERGY,
        kaya_factors.PEFF_per_PEDEq,
        append=False,
    )


def _calc_tfc_per_peff(input_data):
    return input_data.divide(
        kaya_variables.TFC,
        kaya_variables.PRIMARY_ENERGY_FF,
        kaya_factors.TFC_per_PEFF,
        ignore_units="Mt CO2/EJ",
        append=False,
    )


def _calc_nfc_per_tfc(input_data):
    return input_data.divide(
        kaya_variables.NFC,
        kaya_variables.TFC,
        kaya_factors.NFC_per_TFC,
        ignore_units="",
        append=False,
    )

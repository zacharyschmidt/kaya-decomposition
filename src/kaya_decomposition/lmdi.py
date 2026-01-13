"""Compute LMDI (Logarithmic Mean Divisia Index) decomposition."""

import warnings

import numpy as np
import pandas as pd

import pyam
from kaya_decomposition.constants import (
    input_variables,
    kaya_factors as kaya_factor_names,
    kaya_variables as kaya_variable_names,
    lmdi as lmdi_names,
)


# Ordered list of LMDI factor names for iteration
LMDI_FACTOR_NAMES = [
    lmdi_names.Pop_LMDI,
    lmdi_names.GNP_per_P_LMDI,
    lmdi_names.FE_per_GNP_LMDI,
    lmdi_names.PEdeq_per_FE_LMDI,
    lmdi_names.PEFF_per_PEDEq_LMDI,
    lmdi_names.TFC_per_PEFF_LMDI,
]

# Mapping from Kaya factor names to LMDI output names
KAYA_FACTOR_TO_LMDI_NAME = {
    input_variables.POPULATION: lmdi_names.Pop_LMDI,
    kaya_factor_names.GNP_per_P: lmdi_names.GNP_per_P_LMDI,
    kaya_factor_names.FE_per_GNP: lmdi_names.FE_per_GNP_LMDI,
    kaya_factor_names.PEdeq_per_FE: lmdi_names.PEdeq_per_FE_LMDI,
    kaya_factor_names.PEFF_per_PEDEq: lmdi_names.PEFF_per_PEDEq_LMDI,
    kaya_factor_names.TFC_per_PEFF: lmdi_names.TFC_per_PEFF_LMDI,
}


def _logarithmic_mean(a, b):
    """Calculate the logarithmic mean of two values.

    L(a, b) = (a - b) / (ln(a) - ln(b))

    Special cases:
    - If a == b: returns a (by L'Hôpital's rule)
    - If a <= 0 or b <= 0: returns NaN

    Parameters
    ----------
    a, b : float or array-like
        Values to calculate logarithmic mean for.

    Returns
    -------
    float or array-like
        Logarithmic mean.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Suppress expected warnings for edge cases (handled explicitly below)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Handle equal values (limit as b -> a is a)
        result = np.where(
            np.isclose(a, b),
            a,
            (a - b) / (np.log(a) - np.log(b))
        )

        # Handle non-positive values
        result = np.where((a <= 0) | (b <= 0), np.nan, result)

    return result


def compute_lmdi(kaya_factors_df, ref_scenario, int_scenario):
    """Compute corrected LMDI decomposition between two scenarios.

    The LMDI (Logarithmic Mean Divisia Index) method decomposes the difference
    in Total Fossil Carbon (TFC) between a reference and intervention scenario
    into contributions from each Kaya factor.

    Parameters
    ----------
    kaya_factors_df : pyam.IamDataFrame
        Output from compute_kaya_factors() containing data for both scenarios.
    ref_scenario : tuple (model, scenario, region)
        Reference scenario identifiers.
    int_scenario : tuple (model, scenario, region)
        Intervention scenario identifiers.

    Returns
    -------
    pyam.IamDataFrame
        LMDI decomposition results with combined scenario names (ref::int).
        Output variables include Population (LMDI), GNP/P (LMDI), etc.
        The sum of all LMDI terms equals the TFC difference between scenarios.

    Notes
    -----
    The corrected LMDI approach:
    1. Calculate uncorrected LMDI for each factor using logarithmic mean formula
    2. Clip negative values to zero (non-negativity constraint)
    3. Sum non-negative terms to get total
    4. Calculate actual TFC difference between scenarios
    5. Distribute the residual proportionally across factors
    6. Return corrected LMDI values that sum exactly to TFC difference
    """
    ref_input = (
        kaya_factors_df.filter(
            model=ref_scenario[0], scenario=ref_scenario[1], region=ref_scenario[2]
        )
        .as_pandas()
        .assign(scenario_class="reference")
    )
    int_input = (
        kaya_factors_df.filter(
            model=int_scenario[0], scenario=int_scenario[1], region=int_scenario[2]
        )
        .as_pandas()
        .assign(scenario_class="intervention")
    )
    input_data = pyam.IamDataFrame(pd.concat([ref_input, int_input]))

    uncorrected = _uncorrected_lmdi(input_data)
    non_neg = _lmdi_non_neg(uncorrected)
    total_non_neg = _sum_lmdi_non_neg(non_neg)
    total_w_neg = _tfc_diff(input_data)
    difference = total_non_neg.append(total_w_neg).subtract(
        "total_no_neg", "tfc_diff", "difference", append=False, ignore_units=True
    )

    # Apply correction to each LMDI factor
    lmdi_frames = []
    for factor_name in LMDI_FACTOR_NAMES:
        percent = _calc_percent_of_total_for_one_term(
            non_neg, factor_name, total_non_neg
        )
        correction = percent.append(difference).multiply(
            factor_name, "difference", "correction", ignore_units=True
        )
        corrected = correction.append(non_neg).add(
            factor_name, "correction", factor_name, ignore_units=True
        )
        lmdi_frames.append(corrected)

    full_lmdi = pyam.concat(lmdi_frames)
    full_lmdi_no_scenario_class_column = pyam.IamDataFrame(
        full_lmdi.as_pandas().drop(columns="scenario_class")
    )
    return full_lmdi_no_scenario_class_column


def _lmdi_non_neg(uncorrected):
    """Calculate non-negative LMDI terms by clipping negative values to zero."""
    non_neg_frames = [
        _calc_one_non_negative_term(uncorrected, factor_name)
        for factor_name in LMDI_FACTOR_NAMES
    ]
    return pyam.concat(non_neg_frames)


def _sum_lmdi_non_neg(lmdi_non_neg):
    """Sum all non-negative LMDI terms to get total."""
    # Accumulate sum iteratively through factors
    running_sum_name = LMDI_FACTOR_NAMES[0]

    for i, factor_name in enumerate(LMDI_FACTOR_NAMES[1:], start=1):
        is_last = (i == len(LMDI_FACTOR_NAMES) - 1)
        next_sum_name = "total_no_neg" if is_last else f"sum_to_{i}"

        if is_last:
            # Last iteration: return the result without appending
            return lmdi_non_neg.add(
                running_sum_name,
                factor_name,
                next_sum_name,
                append=False,
                ignore_units=True,
            )
        else:
            # Intermediate iterations: append to dataframe and continue
            lmdi_non_neg.add(
                running_sum_name,
                factor_name,
                next_sum_name,
                append=True,
                ignore_units=True,
            )
            running_sum_name = next_sum_name


def _calc_percent_of_total_for_one_term(non_neg, lmdi_term_name, tfc_diff):
    return non_neg.append(tfc_diff).divide(
        lmdi_term_name, "total_no_neg", lmdi_term_name, ignore_units=True
    )


def _tfc_diff(kaya_factors_df):
    (combined_model_name, combined_scenario_name, combined_region_name) = (
        _make_combined_scenario_name(kaya_factors_df.as_pandas())
    )
    tfc = (
        kaya_factors_df.filter(
            variable=kaya_variable_names.TFC, scenario_class="reference"
        )
        .rename(variable={kaya_variable_names.TFC: "tfc_ref"})
        .append(
            kaya_factors_df.filter(
                variable=kaya_variable_names.TFC, scenario_class="intervention"
            )
        )
    )
    tfc = pyam.IamDataFrame(
        tfc.as_pandas()
        .assign(scenario_class="LMDI")
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
    )
    return tfc.subtract(
        "tfc_ref", kaya_variable_names.TFC, "tfc_diff", ignore_units=True
    )


def _calc_one_non_negative_term(uncorrected_lmdi, lmdi_term_name):
    return uncorrected_lmdi.apply(
        _remove_negative, lmdi_term_name, args=[lmdi_term_name], ignore_units=True
    )


def _remove_negative(lmdi_term):
    return lmdi_term.clip(lower=0)


def _uncorrected_lmdi(kaya_factors_df):
    """Calculate uncorrected LMDI terms for all Kaya factors."""
    lmdi_terms = [
        _calc_one_lmdi_term(kaya_factors_df, kaya_factor_name, lmdi_name)
        for kaya_factor_name, lmdi_name in KAYA_FACTOR_TO_LMDI_NAME.items()
    ]
    return pyam.concat(lmdi_terms)


def _calc_one_lmdi_term(
    input_data,
    kaya_factor_name,
    lmdi_term_name,
    kaya_product_name=None,
):
    if kaya_product_name is None:
        kaya_product_name = kaya_variable_names.TFC
    return input_data.apply(
        _lmdi,
        lmdi_term_name,
        axis="variable",
        args=[kaya_factor_name, kaya_product_name],
        ignore_units=True,
    )


def _lmdi(kaya_factor, kaya_product):
    (combined_model_name, combined_scenario_name, combined_region_name) = (
        _make_combined_scenario_name(kaya_factor)
    )

    factor_ref = (
        kaya_factor.reset_index()
        .query('scenario_class == "reference"')
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
        .assign(scenario_class="LMDI")
        .set_index(list(kaya_factor.reset_index().columns[:-1]))
        .rename(columns=lambda x: "value")
    )

    factor_int = (
        kaya_factor.reset_index()
        .query('scenario_class == "intervention"')
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
        .assign(scenario_class="LMDI")
        .set_index(list(kaya_factor.reset_index().columns[:-1]))
        .rename(columns=lambda x: "value")
    )
    tfc_ref = (
        kaya_product.reset_index()
        .query('scenario_class == "reference"')
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
        .assign(scenario_class="LMDI")
        .set_index(list(kaya_factor.reset_index().columns[:-1]))
        .rename(columns=lambda x: "value")
    )
    tfc_int = (
        kaya_product.reset_index()
        .query('scenario_class == "intervention"')
        .assign(
            model=combined_model_name,
            scenario=combined_scenario_name,
            region=combined_region_name,
        )
        .assign(scenario_class="LMDI")
        .set_index(list(kaya_factor.reset_index().columns[:-1]))
        .rename(columns=lambda x: "value")
    )

    # Use logarithmic mean with protection against division by zero
    log_mean = _logarithmic_mean(tfc_ref.values, tfc_int.values)

    # Handle factor ratio with protection against zero/negative values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        factor_ratio = np.where(
            (factor_ref.values > 0) & (factor_int.values > 0),
            np.log(factor_ref.values / factor_int.values),
            0.0
        )

    result = log_mean * factor_ratio

    # Return as Series with the same index structure
    result_df = tfc_ref.copy()
    result_df["value"] = result.flatten()
    return result_df.squeeze(axis=1)


def _make_combined_scenario_name(kaya_factor):
    ref = kaya_factor.reset_index().query('scenario_class == "reference"')
    int_df = kaya_factor.reset_index().query('scenario_class == "intervention"')

    ref_model_name = ref.model.values[0]
    int_model_name = int_df.model.values[0]

    ref_scenario_name = ref.scenario.values[0]
    int_scenario_name = int_df.scenario.values[0]

    ref_region_name = ref.region.values[0]
    int_region_name = int_df.region.values[0]

    return (
        ref_model_name + "::" + int_model_name,
        ref_scenario_name + "::" + int_scenario_name,
        ref_region_name + "::" + int_region_name,
    )

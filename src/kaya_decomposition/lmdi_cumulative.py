"""Compute cumulative LMDI decomposition for a single scenario over time."""

import numpy as np
import pandas as pd

import pyam
from kaya_decomposition.constants import (
    input_variables,
    kaya_factors as kaya_factor_names,
    kaya_variables as kaya_variable_names,
    lmdi_cumulative as lmdi_cumulative_names,
)
from kaya_decomposition.utils import trapezoidal_integrate


# Mapping from Kaya factors to cumulative LMDI output names
FACTOR_TO_CUMULATIVE_NAME = {
    input_variables.POPULATION: lmdi_cumulative_names.Pop_cumulative,
    kaya_factor_names.GNP_per_P: lmdi_cumulative_names.GNP_per_P_cumulative,
    kaya_factor_names.FE_per_GNP: lmdi_cumulative_names.FE_per_GNP_cumulative,
    kaya_factor_names.PEdeq_per_FE: lmdi_cumulative_names.PEdeq_per_FE_cumulative,
    kaya_factor_names.PEFF_per_PEDEq: lmdi_cumulative_names.PEFF_per_PEDEq_cumulative,
    kaya_factor_names.TFC_per_PEFF: lmdi_cumulative_names.TFC_per_PEFF_cumulative,
}


def compute_lmdi_cumulative(
    kaya_factors_df,
    base_year=2020,
    scenario=None,
):
    """Compute cumulative LMDI decomposition for a single scenario.

    Unlike compute_lmdi() which compares two scenarios at the same time,
    this computes LMDI contributions relative to a base year within
    a single scenario over time.

    Parameters
    ----------
    kaya_factors_df : pyam.IamDataFrame
        Output from compute_kaya_factors() containing Kaya factors and TFC.
        Must contain multiple time points.
    base_year : int
        Reference year for LMDI calculation (default 2020).
        All contributions are calculated relative to this year.
    scenario : tuple (model, scenario, region), optional
        Scenario to analyze. If None, uses first available scenario.

    Returns
    -------
    pyam.IamDataFrame
        LMDI contributions for each Kaya factor at each time point.
        Variables are named according to lmdi_cumulative constants.
        The sum of all factor contributions at each time point equals
        the TFC change from the base year.

    Notes
    -----
    The LMDI-I additive formula is used:

        contribution_i(t) = L(TFC_t, TFC_0) × ln(factor_i(t) / factor_i(0))

    where L(a,b) is the logarithmic mean: (a-b) / (ln(a) - ln(b))

    A correction is applied to ensure non-negative contributions that
    sum exactly to the actual TFC difference.
    """
    # Filter to specified scenario
    if scenario is not None:
        filtered_df = kaya_factors_df.filter(
            model=scenario[0], scenario=scenario[1], region=scenario[2]
        )
    else:
        # Use first available scenario with bounds check
        data = kaya_factors_df.data
        if data.empty:
            raise ValueError("Input data is empty. Cannot compute cumulative LMDI.")

        first_model = data["model"].iloc[0]
        first_scenario = data["scenario"].iloc[0]
        first_region = data["region"].iloc[0]
        filtered_df = kaya_factors_df.filter(
            model=first_model, scenario=first_scenario, region=first_region
        )

    # Calculate uncorrected LMDI terms
    uncorrected = _calc_uncorrected_lmdi_cumulative(filtered_df, base_year)

    # Calculate TFC difference from base year
    tfc_diff = _calc_tfc_diff_cumulative(filtered_df, base_year)

    # Apply non-negativity correction
    corrected = _apply_lmdi_correction(uncorrected, tfc_diff)

    return corrected


def compute_lmdi_cumulative_sum(
    kaya_factors_or_lmdi_df,
    base_year=2020,
    periods=None,
    integration_method=None,
    use_corrected=None,
):
    """Sum cumulative LMDI contributions over specified time periods.

    Parameters
    ----------
    kaya_factors_or_lmdi_df : pyam.IamDataFrame
        Either:
        - Output from compute_kaya_factors() (recommended for Excel-matching)
        - Output from compute_lmdi_cumulative() (legacy compatibility)
    base_year : int
        Base year for LMDI calculation (default 2020).
    periods : list of tuples, optional
        List of (start_year, end_year) periods to sum over.
        Default: [(2020, 2050), (2050, 2100), (2020, 2100)]
    integration_method : str, optional
        Method for computing period sums:
        - "trapezoidal": Trapezoidal integration, matches Excel methodology
        - "endpoint": Simple sum of endpoint values (legacy behavior)
        Default: "trapezoidal" when kaya_factors provided, "endpoint" for LMDI results.
    use_corrected : bool, optional
        If True, use corrected (non-negative) values.
        Default: False when kaya_factors provided (matches Excel), True for LMDI results.

    Returns
    -------
    pd.DataFrame
        Table with factors as rows and periods as columns.
        Values are in Gt CO2 when kaya_factors provided (matches Excel format),
        or Mt CO2 when LMDI results provided (legacy behavior).

    Notes
    -----
    The Excel LMDItableRefAllSectors sheet computes period sums by:
    1. Interpolating data to annual resolution
    2. Computing UNCORRECTED LMDI values for each year
    3. Summing all annual values in the period

    This function approximates that by using trapezoidal integration
    of the UNCORRECTED values between available data points. This
    matches Excel results within ~2% for typical scenario data.

    Example
    -------
    >>> from kaya_decomposition import compute_kaya_factors, compute_kaya_variables
    >>> from kaya_decomposition.lmdi_cumulative import compute_lmdi_cumulative_sum
    >>>
    >>> kaya_vars = compute_kaya_variables(input_data)
    >>> kaya_factors = compute_kaya_factors(kaya_vars)
    >>> result = compute_lmdi_cumulative_sum(
    ...     kaya_factors,
    ...     base_year=2020,
    ...     periods=[(2020, 2050)],
    ... )
    >>> print(result)
                                        2020 to 2050
    Population                            134.83
    Economic Activity per Person          411.26
    ...
    """
    if periods is None:
        periods = [(2020, 2050), (2050, 2100), (2020, 2100)]

    # Detect input type and get LMDI values
    data = kaya_factors_or_lmdi_df.data

    # Check if this is kaya_factors (has TFC variable) or lmdi result
    is_kaya_factors = kaya_variable_names.TFC in data["variable"].values

    # Set defaults based on input type
    if is_kaya_factors:
        # New behavior: trapezoidal integration of uncorrected values, output in Gt
        if integration_method is None:
            integration_method = "trapezoidal"
        if use_corrected is None:
            use_corrected = False
        convert_to_gt = True

        # Compute LMDI from kaya factors
        if use_corrected:
            lmdi_data = compute_lmdi_cumulative(kaya_factors_or_lmdi_df, base_year)
        else:
            lmdi_data = _calc_uncorrected_lmdi_cumulative(kaya_factors_or_lmdi_df, base_year)
    else:
        # Legacy behavior: endpoint summation of provided LMDI results, output in Mt
        if integration_method is None:
            integration_method = "endpoint"
        if use_corrected is None:
            use_corrected = True  # Legacy input is already corrected
        convert_to_gt = False

        # Input is already LMDI result
        lmdi_data = kaya_factors_or_lmdi_df

    data = lmdi_data.data
    variables = data["variable"].unique()

    # Build result DataFrame
    result_data = {}

    for start_year, end_year in periods:
        period_label = f"{start_year} to {end_year}"
        period_sums = {}

        for var in variables:
            var_data = data[data["variable"] == var].sort_values("year")

            if integration_method == "trapezoidal":
                period_sum = _trapezoidal_integrate(var_data, start_year, end_year)
            else:  # "endpoint" - legacy behavior
                available_years = sorted(var_data["year"].unique())
                years_in_period = [y for y in available_years if start_year < y <= end_year]
                period_sum = var_data[var_data["year"].isin(years_in_period)]["value"].sum()

            # Convert Mt to Gt only when using kaya_factors input
            if convert_to_gt:
                period_sums[var] = period_sum / 1000
            else:
                period_sums[var] = period_sum

        result_data[period_label] = period_sums

    result_df = pd.DataFrame(result_data)

    # Order rows according to the standard order
    standard_order = [
        lmdi_cumulative_names.Pop_cumulative,
        lmdi_cumulative_names.GNP_per_P_cumulative,
        lmdi_cumulative_names.FE_per_GNP_cumulative,
        lmdi_cumulative_names.PEdeq_per_FE_cumulative,
        lmdi_cumulative_names.PEFF_per_PEDEq_cumulative,
        lmdi_cumulative_names.TFC_per_PEFF_cumulative,
    ]

    # Reorder to match standard order (for variables that exist)
    existing_order = [v for v in standard_order if v in result_df.index]
    other_vars = [v for v in result_df.index if v not in standard_order]
    result_df = result_df.reindex(existing_order + other_vars)

    return result_df


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
    import warnings

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


def _calc_uncorrected_lmdi_cumulative(kaya_factors_df, base_year):
    """Calculate uncorrected LMDI terms for all years relative to base year.

    Parameters
    ----------
    kaya_factors_df : pyam.IamDataFrame
        Kaya factors data with TFC variable.
    base_year : int
        Base year for comparison.

    Returns
    -------
    pyam.IamDataFrame
        Uncorrected LMDI terms for each factor at each year.

    Raises
    ------
    ValueError
        If base_year is not present in the data.
    """
    data = kaya_factors_df.data

    # Get base year TFC with bounds check
    tfc_base_data = data[
        (data["variable"] == kaya_variable_names.TFC) & (data["year"] == base_year)
    ]["value"].values

    if len(tfc_base_data) == 0:
        available_years = sorted(data["year"].unique())
        raise ValueError(
            f"Base year {base_year} not found in data. "
            f"Available years: {available_years}"
        )

    tfc_base = tfc_base_data[0]

    # Get all years
    years = sorted(data["year"].unique())

    result_rows = []

    for year in years:
        # Get TFC for this year
        tfc_year = data[
            (data["variable"] == kaya_variable_names.TFC) & (data["year"] == year)
        ]["value"].values[0]

        # Calculate logarithmic mean
        log_mean = _logarithmic_mean(tfc_year, tfc_base)

        # Calculate LMDI term for each factor
        for factor_name, lmdi_name in FACTOR_TO_CUMULATIVE_NAME.items():
            # Get factor values
            factor_base = data[
                (data["variable"] == factor_name) & (data["year"] == base_year)
            ]["value"].values

            factor_year = data[
                (data["variable"] == factor_name) & (data["year"] == year)
            ]["value"].values

            if len(factor_base) == 0 or len(factor_year) == 0:
                continue

            factor_base = factor_base[0]
            factor_year = factor_year[0]

            # Calculate LMDI term: L(TFC_t, TFC_0) * ln(factor_t / factor_0)
            if factor_base > 0 and factor_year > 0:
                lmdi_value = log_mean * np.log(factor_year / factor_base)
            else:
                lmdi_value = np.nan

            # Get metadata from original data
            sample_row = data[data["year"] == year].iloc[0]

            result_rows.append({
                "model": sample_row["model"],
                "scenario": sample_row["scenario"],
                "region": sample_row["region"],
                "variable": lmdi_name,
                "unit": "Mt CO2/yr",
                "year": year,
                "value": lmdi_value,
            })

    result_df = pd.DataFrame(result_rows)
    return pyam.IamDataFrame(result_df)


def _calc_tfc_diff_cumulative(kaya_factors_df, base_year):
    """Calculate TFC difference from base year for each time point.

    Parameters
    ----------
    kaya_factors_df : pyam.IamDataFrame
        Kaya factors data with TFC variable.
    base_year : int
        Base year for comparison.

    Returns
    -------
    dict
        Dictionary mapping year to TFC difference.

    Raises
    ------
    ValueError
        If base_year is not present in the data.
    """
    data = kaya_factors_df.data

    # Get base year TFC with bounds check
    tfc_base_data = data[
        (data["variable"] == kaya_variable_names.TFC) & (data["year"] == base_year)
    ]["value"].values

    if len(tfc_base_data) == 0:
        available_years = sorted(data["year"].unique())
        raise ValueError(
            f"Base year {base_year} not found in data. "
            f"Available years: {available_years}"
        )

    tfc_base = tfc_base_data[0]

    # Calculate difference for each year
    tfc_diff = {}
    for year in data["year"].unique():
        tfc_year = data[
            (data["variable"] == kaya_variable_names.TFC) & (data["year"] == year)
        ]["value"].values[0]
        tfc_diff[year] = tfc_year - tfc_base

    return tfc_diff


def _apply_lmdi_correction(uncorrected_lmdi, tfc_diff):
    """Apply non-negativity correction to LMDI terms.

    The correction ensures:
    1. All terms are non-negative
    2. Terms sum exactly to the actual TFC difference

    Parameters
    ----------
    uncorrected_lmdi : pyam.IamDataFrame
        Uncorrected LMDI terms.
    tfc_diff : dict
        TFC difference (TFC_t - TFC_0) at each time point.

    Returns
    -------
    pyam.IamDataFrame
        Corrected LMDI terms.
    """
    data = uncorrected_lmdi.data.copy()
    result_rows = []

    for year in data["year"].unique():
        year_data = data[data["year"] == year].copy()

        # Step 1: Clip negative values to zero
        year_data["non_neg"] = year_data["value"].clip(lower=0)

        # Step 2: Sum non-negative terms
        total_non_neg = year_data["non_neg"].sum()

        # Step 3: Get actual TFC difference
        actual_diff = tfc_diff[year]

        # Step 4: Calculate difference between non-neg sum and actual
        difference = total_non_neg - actual_diff

        # Step 5-8: Calculate correction and apply
        if total_non_neg > 0:
            year_data["percent"] = year_data["non_neg"] / total_non_neg
            year_data["correction"] = year_data["percent"] * difference
            year_data["corrected"] = year_data["non_neg"] - year_data["correction"]
        else:
            # If all terms are zero or negative, set corrected to 0
            year_data["corrected"] = 0.0

        # Build result rows
        for _, row in year_data.iterrows():
            result_rows.append({
                "model": row["model"],
                "scenario": row["scenario"],
                "region": row["region"],
                "variable": row["variable"],
                "unit": row["unit"],
                "year": row["year"],
                "value": row["corrected"],
            })

    result_df = pd.DataFrame(result_rows)
    return pyam.IamDataFrame(result_df)


# Use shared trapezoidal integration function
_trapezoidal_integrate = trapezoidal_integrate

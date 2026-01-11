"""Tests for compute_lmdi function."""

import pandas as pd
import pytest
from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal

from kaya_decomposition import (
    compute_kaya_variables,
    compute_kaya_factors,
    compute_lmdi,
    lmdi as lmdi_names,
)


def test_compute_lmdi(two_scenario_dataframe):
    """Test computing LMDI returns correct results."""
    kaya_vars = compute_kaya_variables(two_scenario_dataframe)
    factors = compute_kaya_factors(kaya_vars)
    result = compute_lmdi(
        factors,
        ref_scenario=("model_a", "scen_a", "World"),
        int_scenario=("model_a", "scen_b", "World"),
    )

    expected = IamDataFrame(
        pd.DataFrame(
            [
                [lmdi_names.FE_per_GNP_LMDI, "unknown", 1.321788],
                [lmdi_names.GNP_per_P_LMDI, "unknown", 0],
                [lmdi_names.PEdeq_per_FE_LMDI, "unknown", 0.816780],
                [lmdi_names.PEFF_per_PEDEq_LMDI, "unknown", 0],
                [lmdi_names.Pop_LMDI, "unknown", 0],
                [lmdi_names.TFC_per_PEFF_LMDI, "unknown", 4.853221],
            ],
            columns=["variable", "unit", 2010],
        ),
        model="model_a::model_a",
        scenario="scen_a::scen_b",
        region="World::World",
    )

    assert_iamframe_equal(expected, result)


def test_compute_lmdi_returns_dataframe(two_scenario_dataframe):
    """Test that compute_lmdi returns an IamDataFrame."""
    kaya_vars = compute_kaya_variables(two_scenario_dataframe)
    factors = compute_kaya_factors(kaya_vars)
    result = compute_lmdi(
        factors,
        ref_scenario=("model_a", "scen_a", "World"),
        int_scenario=("model_a", "scen_b", "World"),
    )

    assert isinstance(result, IamDataFrame)


def test_compute_lmdi_has_combined_scenario_names(two_scenario_dataframe):
    """Test that LMDI results have combined scenario names using :: separator."""
    kaya_vars = compute_kaya_variables(two_scenario_dataframe)
    factors = compute_kaya_factors(kaya_vars)
    result = compute_lmdi(
        factors,
        ref_scenario=("model_a", "scen_a", "World"),
        int_scenario=("model_a", "scen_b", "World"),
    )

    # Check combined names
    assert "model_a::model_a" in result.model
    assert "scen_a::scen_b" in result.scenario
    assert "World::World" in result.region


def test_compute_lmdi_has_expected_variables(two_scenario_dataframe):
    """Test that LMDI results contain all expected output variables."""
    kaya_vars = compute_kaya_variables(two_scenario_dataframe)
    factors = compute_kaya_factors(kaya_vars)
    result = compute_lmdi(
        factors,
        ref_scenario=("model_a", "scen_a", "World"),
        int_scenario=("model_a", "scen_b", "World"),
    )

    expected_variables = [
        lmdi_names.Pop_LMDI,
        lmdi_names.GNP_per_P_LMDI,
        lmdi_names.FE_per_GNP_LMDI,
        lmdi_names.PEdeq_per_FE_LMDI,
        lmdi_names.PEFF_per_PEDEq_LMDI,
        lmdi_names.TFC_per_PEFF_LMDI,
    ]

    for var in expected_variables:
        assert var in result.variable, f"Expected variable {var} not found in results"

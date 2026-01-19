"""Tests for compute_kaya_variables function."""

import logging

import pandas as pd
import pytest
from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal

from kaya_decomposition import compute_kaya_variables


def test_compute_kaya_variables(test_dataframe):
    """Test computing kaya variables returns correct results."""
    result = compute_kaya_variables(test_dataframe)

    expected = IamDataFrame(
        pd.DataFrame(
            [
                ["Population", "million", 1000],
                ["GDP|PPP", "billion USD_2005/yr", 6],
                ["Final Energy", "EJ/yr", 8.0],
                ["Primary Energy", "EJ/yr", 10.0],
                ["Primary Energy|Fossil", "EJ/yr", 9.0],
                ["Total Fossil Carbon", "Mt CO2/yr", 12.0],
                ["Net Fossil Carbon", "Mt CO2/yr", 10.0],
            ],
            columns=["variable", "unit", 2010],
        ),
        model="model_a",
        scenario="scen_a",
        region="World",
    )

    assert_iamframe_equal(expected, result)


def test_compute_kaya_variables_raises_on_incomplete_input(test_dataframe):
    """Test that incomplete input raises ValueError with descriptive message."""
    incomplete_df = test_dataframe.filter(variable="Population")
    with pytest.raises(ValueError, match="missing required input variables"):
        compute_kaya_variables(incomplete_df)


def test_compute_kaya_variables_uses_gdp_mer_fallback(test_dataframe):
    """Test GDP|MER is used when GDP|PPP is unavailable."""
    df_no_gdp_ppp = test_dataframe.filter(variable="GDP|PPP", keep=False)
    result = compute_kaya_variables(df_no_gdp_ppp)

    # Result should contain GDP|MER instead of GDP|PPP
    assert "GDP|MER" in result.variable
    assert "GDP|PPP" not in result.variable


def test_compute_kaya_variables_raises_when_no_gdp(test_dataframe):
    """Test that ValueError is raised when both GDP variants are unavailable."""
    df_no_gdp = test_dataframe.filter(
        variable=["GDP|PPP", "GDP|MER"],
        keep=False,
    )
    with pytest.raises(ValueError, match="missing required input variables"):
        compute_kaya_variables(df_no_gdp)


def test_compute_kaya_variables_logs_missing_variables(test_dataframe, caplog):
    """Test that missing variables are logged before raising ValueError."""
    df_no_pop = test_dataframe.filter(variable="Population", keep=False)

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError, match="missing required input variables"):
            compute_kaya_variables(df_no_pop)

    # Check that the log message contains expected information
    assert "model: model_a" in caplog.text
    assert "scenario: scen_a" in caplog.text
    assert "region: World" in caplog.text
    assert "Population" in caplog.text

"""Tests for compute_kaya_factors function."""

import pandas as pd
import pytest
from pyam import IamDataFrame
from pyam.testing import assert_iamframe_equal

from kaya_decomposition import compute_kaya_variables, compute_kaya_factors


def test_compute_kaya_factors(test_dataframe):
    """Test computing kaya factors returns correct results."""
    kaya_vars = compute_kaya_variables(test_dataframe)
    result = compute_kaya_factors(kaya_vars)

    expected = IamDataFrame(
        pd.DataFrame(
            [
                ["FE/GNP", "EJ / USD / billion", 1.33333],
                ["GNP/P", "USD * billion / million / a", 0.006000],
                ["NFC/TFC", "", 0.833333],
                ["PEDEq/FE", "", 1.250000],
                ["PEFF/PEDEq", "", 0.900000],
                ["TFC/PEFF", "Mt CO2/EJ", 1.333333],
                ["Population", "million", 1000],
                ["Total Fossil Carbon", "Mt CO2/yr", 12.0],
            ],
            columns=["variable", "unit", 2010],
        ),
        model="model_a",
        scenario="scen_a",
        region="World",
    )

    assert_iamframe_equal(expected, result)


def test_compute_kaya_factors_raises_on_none_input():
    """Test that None input raises ValueError with descriptive message."""
    with pytest.raises(ValueError, match="kaya_variables_frame is None"):
        compute_kaya_factors(None)


def test_compute_kaya_factors_raises_on_wrong_type():
    """Test that wrong input type raises ValueError."""
    with pytest.raises(ValueError, match="expected pyam.IamDataFrame"):
        compute_kaya_factors("not a dataframe")


def test_compute_kaya_factors_raises_on_empty_dataframe():
    """Test that empty IamDataFrame raises ValueError."""
    empty_df = IamDataFrame(
        pd.DataFrame(columns=["variable", "unit", 2010]),
        model="test",
        scenario="test",
        region="World",
    )
    with pytest.raises(ValueError, match="input IamDataFrame is empty"):
        compute_kaya_factors(empty_df)


def test_compute_kaya_variables_raises_before_factors_on_incomplete(test_dataframe):
    """Test that incomplete input raises ValueError in compute_kaya_variables."""
    incomplete_df = test_dataframe.filter(variable="Population")
    with pytest.raises(ValueError, match="missing required input variables"):
        compute_kaya_variables(incomplete_df)


def test_compute_kaya_factors_uses_gdp_mer_fallback(test_dataframe):
    """Test GDP|MER is used for factor calculations when GDP|PPP unavailable."""
    df_no_gdp_ppp = test_dataframe.filter(variable="GDP|PPP", keep=False)
    kaya_vars = compute_kaya_variables(df_no_gdp_ppp)
    result = compute_kaya_factors(kaya_vars)

    # Create expected result using GDP|MER instead of GDP|PPP for calculations
    expected = IamDataFrame(
        pd.DataFrame(
            [
                # 8 EJ / 5 billion USD = 1.6
                ["FE/GNP", "EJ / USD / billion", 1.6],
                # 5 billion USD / 1000 million = 0.005
                ["GNP/P", "USD * billion / million / a", 0.005],
                ["NFC/TFC", "", 0.833333],
                ["PEDEq/FE", "", 1.250000],
                ["PEFF/PEDEq", "", 0.900000],
                ["TFC/PEFF", "Mt CO2/EJ", 1.333333],
                ["Population", "million", 1000],
                ["Total Fossil Carbon", "Mt CO2/yr", 12.0],
            ],
            columns=["variable", "unit", 2010],
        ),
        model="model_a",
        scenario="scen_a",
        region="World",
    )

    assert_iamframe_equal(expected, result)

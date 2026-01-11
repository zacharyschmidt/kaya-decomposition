"""Pytest configuration and shared fixtures."""

import pandas as pd
import pytest
from pyam import IamDataFrame


@pytest.fixture
def test_dataframe():
    """Create a test IamDataFrame with all required input variables."""
    return IamDataFrame(
        pd.DataFrame(
            [
                ["Population", "million", 1000],
                ["GDP|PPP", "billion USD_2005/yr", 6],
                ["GDP|MER", "billion USD_2005/yr", 5],
                ["Final Energy", "EJ/yr", 8],
                ["Primary Energy", "EJ/yr", 10],
                ["Primary Energy|Coal", "EJ/yr", 5],
                ["Primary Energy|Gas", "EJ/yr", 2],
                ["Primary Energy|Oil", "EJ/yr", 2],
                ["Emissions|CO2|Fossil Fuels and Industry", "Mt CO2/yr", 10],
                ["Emissions|CO2|Industrial Processes", "Mt CO2/yr", 1],
                ["Emissions|CO2|AFOLU", "Mt CO2/yr", 1],
                ["Emissions|CO2|Carbon Capture and Storage", "Mt CO2/yr", 4],
                ["Emissions|CO2|Carbon Capture and Storage|Biomass", "Mt CO2/yr", 1],
                ["Carbon Sequestration|CCS|Fossil|Energy", "Mt CO2/yr", 2],
                ["Carbon Sequestration|CCS|Fossil|Industrial Processes", "Mt CO2/yr", 1],
                ["Carbon Sequestration|CCS|Biomass|Energy", "Mt CO2/yr", 0.5],
                ["Carbon Sequestration|CCS|Biomass|Industrial Processes", "Mt CO2/yr", 0.5],
            ],
            columns=["variable", "unit", 2010],
        ),
        model="model_a",
        scenario="scen_a",
        region="World",
    )


@pytest.fixture
def two_scenario_dataframe():
    """Create a test IamDataFrame with two scenarios for LMDI testing."""
    scen_a = IamDataFrame(
        pd.DataFrame(
            [
                ["Population", "million", 1000],
                ["GDP|PPP", "billion USD_2005/yr", 6],
                ["GDP|MER", "billion USD_2005/yr", 5],
                ["Final Energy", "EJ/yr", 8],
                ["Primary Energy", "EJ/yr", 10],
                ["Primary Energy|Coal", "EJ/yr", 5],
                ["Primary Energy|Gas", "EJ/yr", 2],
                ["Primary Energy|Oil", "EJ/yr", 2],
                ["Emissions|CO2|Fossil Fuels and Industry", "Mt CO2/yr", 10],
                ["Emissions|CO2|Industrial Processes", "Mt CO2/yr", 1],
                ["Emissions|CO2|AFOLU", "Mt CO2/yr", 1],
                ["Emissions|CO2|Carbon Capture and Storage", "Mt CO2/yr", 4],
                ["Emissions|CO2|Carbon Capture and Storage|Biomass", "Mt CO2/yr", 1],
                ["Carbon Sequestration|CCS|Fossil|Energy", "Mt CO2/yr", 2],
                ["Carbon Sequestration|CCS|Fossil|Industrial Processes", "Mt CO2/yr", 1],
                ["Carbon Sequestration|CCS|Biomass|Energy", "Mt CO2/yr", 0.5],
                ["Carbon Sequestration|CCS|Biomass|Industrial Processes", "Mt CO2/yr", 0.5],
            ],
            columns=["variable", "unit", 2010],
        ),
        model="model_a",
        scenario="scen_a",
        region="World",
    )
    scen_b = IamDataFrame(
        pd.DataFrame(
            [
                ["Population", "million", 1001],
                ["GDP|PPP", "billion USD_2005/yr", 7],
                ["GDP|MER", "billion USD_2005/yr", 6],
                ["Final Energy", "EJ/yr", 9],
                ["Primary Energy", "EJ/yr", 11],
                ["Primary Energy|Coal", "EJ/yr", 6],
                ["Primary Energy|Gas", "EJ/yr", 3],
                ["Primary Energy|Oil", "EJ/yr", 3],
                ["Emissions|CO2|Fossil Fuels and Industry", "Mt CO2/yr", 13],
                ["Emissions|CO2|Industrial Processes", "Mt CO2/yr", 2],
                ["Emissions|CO2|AFOLU", "Mt CO2/yr", 2],
                ["Emissions|CO2|Carbon Capture and Storage", "Mt CO2/yr", 5],
                ["Emissions|CO2|Carbon Capture and Storage|Biomass", "Mt CO2/yr", 2],
                ["Carbon Sequestration|CCS|Fossil|Energy", "Mt CO2/yr", 3],
                ["Carbon Sequestration|CCS|Fossil|Industrial Processes", "Mt CO2/yr", 2],
                ["Carbon Sequestration|CCS|Biomass|Energy", "Mt CO2/yr", 1.5],
                ["Carbon Sequestration|CCS|Biomass|Industrial Processes", "Mt CO2/yr", 1.5],
            ],
            columns=["variable", "unit", 2010],
        ),
        model="model_a",
        scenario="scen_b",
        region="World",
    )
    return scen_a.append(scen_b)

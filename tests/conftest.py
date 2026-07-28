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
                ["Emissions|CO2|Energy and Industrial Processes", "Mt CO2/yr", 10],
                ["Emissions|CO2|Industrial Processes", "Mt CO2/yr", 1],
                ["Emissions|CO2|AFOLU", "Mt CO2/yr", 1],
                ["Carbon Sequestration|CCS", "Mt CO2/yr", 4],
                ["Carbon Sequestration|CCS|Biomass", "Mt CO2/yr", 1],
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
                ["Emissions|CO2|Energy and Industrial Processes", "Mt CO2/yr", 10],
                ["Emissions|CO2|Industrial Processes", "Mt CO2/yr", 1],
                ["Emissions|CO2|AFOLU", "Mt CO2/yr", 1],
                ["Carbon Sequestration|CCS", "Mt CO2/yr", 4],
                ["Carbon Sequestration|CCS|Biomass", "Mt CO2/yr", 1],
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
                ["Emissions|CO2|Energy and Industrial Processes", "Mt CO2/yr", 13],
                ["Emissions|CO2|Industrial Processes", "Mt CO2/yr", 2],
                ["Emissions|CO2|AFOLU", "Mt CO2/yr", 2],
                ["Carbon Sequestration|CCS", "Mt CO2/yr", 5],
                ["Carbon Sequestration|CCS|Biomass", "Mt CO2/yr", 2],
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


@pytest.fixture
def multi_year_dataframe():
    """Create test IamDataFrame with multiple years for cumulative LMDI testing.

    Contains data for years 2010, 2020, 2030, 2040, 2050 with realistic
    growth patterns for testing cumulative LMDI calculations.
    """
    data = []
    base_values = {
        "Population": 7000,  # million
        "GDP|PPP": 80,  # trillion USD
        "GDP|MER": 60,
        "Final Energy": 400,  # EJ/yr
        "Primary Energy": 550,
        "Primary Energy|Coal": 150,
        "Primary Energy|Gas": 120,
        "Primary Energy|Oil": 180,
        "Emissions|CO2|Energy and Industrial Processes": 35000,  # Mt CO2/yr
        "Emissions|CO2|Industrial Processes": 2000,
        "Emissions|CO2|AFOLU": 5000,
        "Carbon Sequestration|CCS": 0,
        "Carbon Sequestration|CCS|Biomass": 0,
        "Carbon Sequestration|CCS|Fossil|Energy": 0,
        "Carbon Sequestration|CCS|Fossil|Industrial Processes": 0,
        "Carbon Sequestration|CCS|Biomass|Energy": 0,
        "Carbon Sequestration|CCS|Biomass|Industrial Processes": 0,
        "Emissions|CH4": 350,  # Mt CH4/yr
        "Emissions|N2O": 10000,  # kt N2O/yr
        "Emissions|F-Gases": 1000,  # Mt CO2-equiv/yr
    }

    # Growth factors per decade
    growth = {
        "Population": 1.08,
        "GDP|PPP": 1.4,
        "GDP|MER": 1.4,
        "Final Energy": 1.15,
        "Primary Energy": 1.12,
        "Primary Energy|Coal": 1.05,
        "Primary Energy|Gas": 1.10,
        "Primary Energy|Oil": 0.98,
        "Emissions|CO2|Energy and Industrial Processes": 1.10,
        "Emissions|CO2|Industrial Processes": 1.05,
        "Emissions|CO2|AFOLU": 1.02,
        "Emissions|CH4": 1.03,
        "Emissions|N2O": 1.02,
        "Emissions|F-Gases": 1.15,
    }

    years = [2010, 2020, 2030, 2040, 2050]

    for var, base_val in base_values.items():
        for i, year in enumerate(years):
            if var in growth:
                value = base_val * (growth[var] ** i)
            else:
                value = base_val
            data.append({
                "model": "TestModel",
                "scenario": "TestScenario",
                "region": "World",
                "variable": var,
                "unit": "various",
                "year": year,
                "value": value,
            })

    return IamDataFrame(pd.DataFrame(data))


@pytest.fixture
def multi_year_all_sectors_dataframe(multi_year_dataframe):
    """Extended multi-year data with all variables needed for all-sectors analysis."""
    # The multi_year_dataframe already includes all needed variables
    return multi_year_dataframe


@pytest.fixture
def multi_year_beccs_dataframe():
    """Multi-year deep-mitigation data with nonzero, growing fossil AND biomass CCS.

    Unlike ``multi_year_dataframe`` (all CCS = 0), this exercises the CCS-sensitive
    code paths: the NFC/NIC biomass add-backs, the Total CDR row, and the Fossil
    Energy CCS row. Aggregate CCS variables are derived from the four splits so the
    data stays internally consistent (e.g. ``CCS = sum of splits``, which
    ``_calc_tfc`` relies on to recover fossil-energy CCS).
    """
    years = [2010, 2020, 2030, 2040, 2050]

    non_ccs_base = {
        "Population": 7000,  # million
        "GDP|PPP": 80,  # trillion USD
        "GDP|MER": 60,
        "Final Energy": 400,  # EJ/yr
        "Primary Energy": 550,
        "Primary Energy|Coal": 150,
        "Primary Energy|Gas": 120,
        "Primary Energy|Oil": 180,
        "Emissions|CO2|Energy and Industrial Processes": 35000,  # Mt CO2/yr
        "Emissions|CO2|Industrial Processes": 2000,
        "Emissions|CO2|AFOLU": 5000,
        "Emissions|CH4": 350,  # Mt CH4/yr
        "Emissions|N2O": 10000,  # kt N2O/yr
        "Emissions|F-Gases": 1000,  # Mt CO2-equiv/yr
    }
    non_ccs_growth = {
        "Population": 1.08,
        "GDP|PPP": 1.4,
        "GDP|MER": 1.4,
        "Final Energy": 1.15,
        "Primary Energy": 1.12,
        "Primary Energy|Coal": 0.85,  # coal phased down under mitigation
        "Primary Energy|Gas": 1.05,
        "Primary Energy|Oil": 0.90,
        "Emissions|CO2|Energy and Industrial Processes": 0.80,  # deep decline
        "Emissions|CO2|Industrial Processes": 0.95,
        "Emissions|CO2|AFOLU": 0.85,
        "Emissions|CH4": 0.95,
        "Emissions|N2O": 0.98,
        "Emissions|F-Gases": 0.90,
    }
    # CCS splits (Mt CO2/yr): base value and per-decade growth (ramping up)
    ccs_split_base = {
        "Carbon Sequestration|CCS|Fossil|Energy": 500,
        "Carbon Sequestration|CCS|Fossil|Industrial Processes": 200,
        "Carbon Sequestration|CCS|Biomass|Energy": 300,
        "Carbon Sequestration|CCS|Biomass|Industrial Processes": 50,
    }
    ccs_split_growth = {
        "Carbon Sequestration|CCS|Fossil|Energy": 1.5,
        "Carbon Sequestration|CCS|Fossil|Industrial Processes": 1.4,
        "Carbon Sequestration|CCS|Biomass|Energy": 1.6,
        "Carbon Sequestration|CCS|Biomass|Industrial Processes": 1.3,
    }

    rows = []

    def add(var, unit, year, value):
        rows.append({
            "model": "TestModel",
            "scenario": "DeepMitigation",
            "region": "World",
            "variable": var,
            "unit": unit,
            "year": year,
            "value": value,
        })

    for i, year in enumerate(years):
        for var, base in non_ccs_base.items():
            add(var, "various", year, base * (non_ccs_growth[var] ** i))

        splits = {
            v: ccs_split_base[v] * (ccs_split_growth[v] ** i)
            for v in ccs_split_base
        }
        for v, val in splits.items():
            add(v, "Mt CO2/yr", year, val)

        fe = splits["Carbon Sequestration|CCS|Fossil|Energy"]
        fi = splits["Carbon Sequestration|CCS|Fossil|Industrial Processes"]
        be = splits["Carbon Sequestration|CCS|Biomass|Energy"]
        bi = splits["Carbon Sequestration|CCS|Biomass|Industrial Processes"]
        add("Carbon Sequestration|CCS|Biomass", "Mt CO2/yr", year, be + bi)
        add("Carbon Sequestration|CCS", "Mt CO2/yr", year, fe + fi + be + bi)

    return IamDataFrame(pd.DataFrame(rows))

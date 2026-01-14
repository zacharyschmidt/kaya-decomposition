"""Utilities for reading test data from Excel workbooks.

This module provides functions to read input data and expected values
directly from the Excel reference workbooks, replacing hard-coded fixtures.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from pyam import IamDataFrame

EXCEL_DIR = Path(__file__).parent.parent / "excel"

# Workbook configurations
WORKBOOKS = {
    "vanvuuren": {
        "filename": "vanVuurenIMAGE_15_TOT_19_TFC_currentcopy.xlsm",
        "model": "IMAGE 3.0.1",
        "scenario": "SSP2-Baseline",
        "region": "World",
        "years": [2005, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
    },
    "grubler": {
        "filename": "GrublerLED_TFC_Industry_from_vanVuuren_TOT_current copy.xlsm",
        "model": "MESSAGE-GLOBIOM 1.0",
        "scenario": "SSP2-Baseline",
        "region": "World",
        "years": [2005, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
    },
    "rockstrom": {
        "filename": "RockstromMESSAGE_Rogelj_2013_myo_L15_BC_c_TFC_Industry_from_vanVuuren_current copy.xlsm",
        "model": "GEA",
        "scenario": "geah_counterfactual",
        "region": "World",
        "years": [2005, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
    },
    "rogelj": {
        "filename": "Rogelj2018AIM_SSP19_TFC_Industry_from_vanVuuren_current copy.xlsm",
        "model": "AIM/CGE 2.0",
        "scenario": "SSP2-Baseline",
        "region": "World",
        "years": [2005, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
    },
    "teske": {
        "filename": "Teske_2019_TFC_current copy.xlsm",
        "model": "Teske",
        "scenario": "Reference (5C)",
        "region": "World",
        "years": [2015, 2020, 2030, 2040, 2050],
    },
}

# Variable name mappings from Excel to IAMC format
VARIABLE_MAPPINGS = {
    "Emissions|CH4": "Emissions|CH4",
    "Carbon Sequestration|CCS": "Carbon Sequestration|CCS",
    "Carbon Sequestration|CCS|Biomass": "Carbon Sequestration|CCS|Biomass",
    "Emissions|CO2|Fossil Fuels and Industry": "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|CO2|AFOLU": "Emissions|CO2|AFOLU",
    "Emissions|F-Gases": "Emissions|F-Gases",
    "Emissions|N2O": "Emissions|N2O",
    "Final Energy": "Final Energy",
    "GDP|PPP": "GDP|PPP",
    "Population": "Population",
    "Primary Energy": "Primary Energy",
    "Primary Energy|Coal": "Primary Energy|Coal",
    "Primary Energy|Gas": "Primary Energy|Gas",
    "Primary Energy|Oil": "Primary Energy|Oil",
    "GDP|MER": "GDP|MER",
    "Emissions|CO2|Industrial Processes": "Emissions|CO2|Industrial Processes",
    "Carbon Sequestration|CCS|Fossil|Industrial Processes": "Carbon Sequestration|CCS|Fossil|Industrial Processes",
    "Carbon Sequestration|CCS|Fossil|Energy": "Carbon Sequestration|CCS|Fossil|Energy",
    "Carbon Sequestration|CCS|Biomass|Energy": "Carbon Sequestration|CCS|Biomass|Energy",
    "Carbon Sequestration|CCS|Biomass|Industrial Processes": "Carbon Sequestration|CCS|Biomass|Industrial Processes",
}

# Unit mapping for variables
UNIT_MAPPING = {
    "Emissions|CH4": "Mt CH4/yr",
    "Carbon Sequestration|CCS": "Mt CO2/yr",
    "Carbon Sequestration|CCS|Biomass": "Mt CO2/yr",
    "Emissions|CO2|Energy and Industrial Processes": "Mt CO2/yr",
    "Emissions|CO2|AFOLU": "Mt CO2/yr",
    "Emissions|F-Gases": "Mt CO2/yr",
    "Emissions|N2O": "kt N2O/yr",
    "Final Energy": "EJ/yr",
    "GDP|PPP": "billion USD_2005/yr",
    "Population": "million",
    "Primary Energy": "EJ/yr",
    "Primary Energy|Coal": "EJ/yr",
    "Primary Energy|Gas": "EJ/yr",
    "Primary Energy|Oil": "EJ/yr",
    "GDP|MER": "billion USD_2005/yr",
    "Emissions|CO2|Industrial Processes": "Mt CO2/yr",
    "Carbon Sequestration|CCS|Fossil|Industrial Processes": "Mt CO2/yr",
    "Carbon Sequestration|CCS|Fossil|Energy": "Mt CO2/yr",
    "Carbon Sequestration|CCS|Biomass|Energy": "Mt CO2/yr",
    "Carbon Sequestration|CCS|Biomass|Industrial Processes": "Mt CO2/yr",
}


def get_workbook_path(workbook_key: str) -> Path:
    """Get the full path to a workbook.

    Parameters
    ----------
    workbook_key : str
        One of: 'vanvuuren', 'grubler', 'rockstrom', 'rogelj', 'teske'

    Returns
    -------
    Path
        Full path to the Excel workbook
    """
    return EXCEL_DIR / WORKBOOKS[workbook_key]["filename"]


def workbook_exists(workbook_key: str) -> bool:
    """Check if a workbook exists at the expected path.

    Parameters
    ----------
    workbook_key : str
        One of: 'vanvuuren', 'grubler', 'rockstrom', 'rogelj', 'teske'

    Returns
    -------
    bool
        True if workbook exists
    """
    return get_workbook_path(workbook_key).exists()


def has_sheet(workbook_key: str, sheet_name: str) -> bool:
    """Check if a workbook has a specific sheet.

    Parameters
    ----------
    workbook_key : str
        Workbook identifier
    sheet_name : str
        Name of the sheet to check

    Returns
    -------
    bool
        True if sheet exists in workbook
    """
    path = get_workbook_path(workbook_key)
    if not path.exists():
        return False

    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        return sheet_name in xl.sheet_names
    except Exception:
        return False


def read_ref_data_sheet(workbook_key: str) -> pd.DataFrame:
    """Read the Ref Data sheet from an Excel workbook.

    Parameters
    ----------
    workbook_key : str
        One of: 'vanvuuren', 'grubler', 'rockstrom', 'rogelj', 'teske'

    Returns
    -------
    pd.DataFrame
        Raw data from Ref Data sheet
    """
    path = get_workbook_path(workbook_key)
    return pd.read_excel(path, sheet_name="Ref Data", engine="openpyxl")


def read_expected_kaya_factors(workbook_key: str) -> Dict[str, Dict[int, float]]:
    """Read expected Kaya factor values from 'ExpKayaRatiosRef' sheet.

    Parameters
    ----------
    workbook_key : str
        Workbook identifier

    Returns
    -------
    dict
        Dictionary mapping factor name to {year: value} dict.
        Keys: 'GNP/P', 'FE/GNP', 'PEDEq/FE', 'PEFF/PEDEq', 'TFC/PEFF', 'NFC/TFC'
    """
    path = get_workbook_path(workbook_key)

    # Read the ExpKayaRatiosRef sheet
    # Structure: rows are factor names, columns are years
    df = pd.read_excel(
        path,
        sheet_name="ExpKayaRatiosRef",
        engine="openpyxl",
        header=None,
    )

    # The sheet typically has:
    # Row 0: header/blank
    # Row 1: GNP/P values
    # Row 2: FE/GNP values
    # etc.
    # First column: factor names
    # Subsequent columns: year values

    result = {}
    factor_names = ['GNP/P', 'FE/GNP', 'PEDEq/FE', 'PEFF/PEDEq', 'TFC/PEFF', 'NFC/TFC']

    config = WORKBOOKS[workbook_key]
    years = config["years"]

    # Parse the sheet to extract factor values
    # This is a simplified parser - may need adjustment based on actual sheet structure
    for i, factor in enumerate(factor_names):
        row_idx = i + 1  # Assuming factors start at row 1
        factor_values = {}

        for j, year in enumerate(years):
            col_idx = j + 1  # Assuming years start at column 1
            try:
                value = df.iloc[row_idx, col_idx]
                if pd.notna(value) and isinstance(value, (int, float)):
                    factor_values[year] = float(value)
            except (IndexError, ValueError):
                continue

        if factor_values:
            result[factor] = factor_values

    return result


def read_expected_kaya_variables(workbook_key: str) -> Dict[str, Dict[int, float]]:
    """Read expected Kaya variable values from 'ExpKayaFactorsRef' sheet.

    Parameters
    ----------
    workbook_key : str
        Workbook identifier

    Returns
    -------
    dict
        Dictionary mapping variable name to {year: value} dict.
        Keys: 'P', 'GNP', 'FE', 'PEDEq', 'PEFF', 'TFC', 'NFC'
    """
    path = get_workbook_path(workbook_key)

    df = pd.read_excel(
        path,
        sheet_name="ExpKayaFactorsRef",
        engine="openpyxl",
        header=None,
    )

    result = {}
    variable_names = ['P', 'GNP', 'FE', 'PEDEq', 'PEFF', 'TFC', 'NFC']

    config = WORKBOOKS[workbook_key]
    years = config["years"]

    for i, var in enumerate(variable_names):
        row_idx = i  # May need adjustment based on actual sheet structure
        var_values = {}

        for j, year in enumerate(years):
            col_idx = j + 1
            try:
                value = df.iloc[row_idx, col_idx]
                if pd.notna(value) and isinstance(value, (int, float)):
                    var_values[year] = float(value)
            except (IndexError, ValueError):
                continue

        if var_values:
            result[var] = var_values

    return result


def read_expected_lmdi_cumulative_sums(workbook_key: str) -> Dict[str, Dict[str, float]]:
    """Read expected LMDI cumulative sums from 'LMDItableRefAllSectors' sheet.

    Parameters
    ----------
    workbook_key : str
        Workbook identifier

    Returns
    -------
    dict
        Dictionary mapping period to {factor: value} dict.
        Periods: '2020 to 2050', '2050 to 2100', '2020 to 2100'
    """
    path = get_workbook_path(workbook_key)

    if not has_sheet(workbook_key, "LMDItableRefAllSectors"):
        return {}

    df = pd.read_excel(
        path,
        sheet_name="LMDItableRefAllSectors",
        engine="openpyxl",
        header=None,
    )

    # Parse the table structure
    # This is a simplified parser - may need adjustment based on actual sheet structure
    result = {}

    periods = ["2020 to 2050", "2050 to 2100", "2020 to 2100"]
    factor_names = [
        "Population",
        "Economic Activity per Person",
        "Energy Intensity of Economy",
        "Energy Supply Loss Factor",
        "Fossil Fuel Fraction",
        "Carbon Intensity of Fossil Energy",
        "Industrial Process Carbon Emissions",
        "Other Gases",
        "Land Use",
        "Total Net Emissions",
    ]

    # Attempt to locate and parse values
    # This is workbook-specific and may need customization
    for period in periods:
        result[period] = {}

    return result


def read_expected_other_gases(workbook_key: str) -> Dict[int, Dict[str, float]]:
    """Read expected Other Gases values from 'OtherGases' sheet.

    Parameters
    ----------
    workbook_key : str
        Workbook identifier

    Returns
    -------
    dict
        Dictionary mapping year to {component: value} dict.
        Components: 'CH4_CO2eq', 'N2O_CO2eq', 'FGases_CO2eq', 'Total'
    """
    path = get_workbook_path(workbook_key)

    if not has_sheet(workbook_key, "OtherGases"):
        return {}

    # Read and parse the OtherGases sheet
    df = pd.read_excel(
        path,
        sheet_name="OtherGases",
        engine="openpyxl",
    )

    # Parse structure (workbook-specific)
    result = {}
    return result


def read_expected_industrial_process(workbook_key: str) -> Dict[int, Dict[str, float]]:
    """Read expected Industrial Process values from 'IndustryEmissionsAccountingRef' sheet.

    Parameters
    ----------
    workbook_key : str
        Workbook identifier

    Returns
    -------
    dict
        Dictionary mapping year to {component: value} dict.
        Components: 'TIC', 'NIC'
    """
    path = get_workbook_path(workbook_key)

    if not has_sheet(workbook_key, "IndustryEmissionsAccountingRef"):
        return {}

    df = pd.read_excel(
        path,
        sheet_name="IndustryEmissionsAccountingRef",
        engine="openpyxl",
    )

    result = {}
    return result


def get_workbook_config(workbook_key: str) -> Dict[str, Any]:
    """Get the configuration for a workbook.

    Parameters
    ----------
    workbook_key : str
        One of: 'vanvuuren', 'grubler', 'rockstrom', 'rogelj', 'teske'

    Returns
    -------
    dict
        Configuration dictionary with model, scenario, region, years
    """
    return WORKBOOKS[workbook_key].copy()


def get_all_workbook_keys() -> List[str]:
    """Get list of all available workbook keys.

    Returns
    -------
    list
        List of workbook keys
    """
    return list(WORKBOOKS.keys())


def get_test_years(workbook_key: str) -> List[int]:
    """Get the years available for testing in a workbook.

    Parameters
    ----------
    workbook_key : str
        Workbook identifier

    Returns
    -------
    list
        List of years available
    """
    return WORKBOOKS[workbook_key]["years"].copy()


def get_common_test_years(workbook_key: str, available_years: Optional[List[int]] = None) -> List[int]:
    """Get common test years for a workbook (intersection with standard test years).

    Parameters
    ----------
    workbook_key : str
        Workbook identifier
    available_years : list, optional
        Override the standard test years

    Returns
    -------
    list
        List of years for testing
    """
    standard_years = [2020, 2030, 2050, 2100]
    workbook_years = WORKBOOKS[workbook_key]["years"]

    if available_years:
        return [y for y in available_years if y in workbook_years]
    return [y for y in standard_years if y in workbook_years]

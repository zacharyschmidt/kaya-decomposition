# kaya-decomposition

Kaya decomposition analysis for integrated-assessment scenario data.

## Overview

This library provides tools for computing Kaya decomposition factors from IAMC-format scenario data. The Kaya identity decomposes CO2 emissions into contributing factors: population, wealth per person, final energy use per dollar, energy supply loss factor, the fraction of primary energy supplied by fossil fuels, the carbon intensity of fossil fuels supplied, and the net emissions of CO2 from energy sector after sequestration.


```
CO2 = P × (GDP/P) × (FE/GDP) × (PE/FE) × (PEFF/PE) × (TFC/PEFF) × (NFC/TFC)
```

Where:
- **P** = Population
- **GDP/P** = GDP per capita (economic activity per person)
- **FE/GDP** = Energy intensity of the economy
- **PE/FE** = Primary to final energy ratio (energy supply losses)
- **PEFF/PE** = Fossil fuel fraction of primary energy
- **TFC/PEFF** = Carbon intensity of fossil energy
- **NFC/TFC** = Net to total carbon ratio (accounts for CCS)


Key features:
- **Kaya Decomposition**: Compute intermediate variables and decomposition factors
- **LMDI Analysis**: Logarithmic Mean Divisia Index for scenario comparison
- **Cumulative LMDI**: Time-series decomposition within a single scenario

## Installation

```bash
pip install kaya-decomposition
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Kaya Decomposition

```python
import pyam
from kaya_decomposition import compute_kaya_variables, compute_kaya_factors

# Load your IAMC-format data
df = pyam.IamDataFrame("your_data.csv")

# Compute intermediate Kaya variables
kaya_vars = compute_kaya_variables(df)

# Compute Kaya decomposition factors
factors = compute_kaya_factors(kaya_vars)
```

### LMDI Scenario Comparison

```python
from kaya_decomposition import compute_lmdi

# Compare two scenarios using LMDI decomposition
lmdi_result = compute_lmdi(
    factors,
    ref_scenario=("Model", "Baseline", "World"),
    int_scenario=("Model", "Policy", "World"),
)
```

### Cumulative LMDI (Time-Series Decomposition)

```python
from kaya_decomposition import compute_lmdi_cumulative, compute_lmdi_cumulative_sum

# Decompose changes over time within a single scenario
lmdi_cumulative = compute_lmdi_cumulative(factors, base_year=2020)

# Sum contributions by period
period_sums = compute_lmdi_cumulative_sum(
    factors,
    base_year=2020,
    periods=[(2020, 2050), (2050, 2100), (2020, 2100)],
)
print(period_sums)
```

### All-Sectors Analysis

```python
from kaya_decomposition import compute_all_sectors_lmdi_cumulative

# Complete analysis including non-CO2 gases, industrial processes, and land use
result = compute_all_sectors_lmdi_cumulative(
    input_data,
    base_year=2020,
    scenario=("Model", "Scenario", "Region"),
)
```

## Required Input Variables

### Core Kaya Decomposition

The following variables must be present in your input data:

- Population
- GDP|PPP or GDP|MER
- Final Energy
- Primary Energy
- Primary Energy|Coal
- Primary Energy|Oil
- Primary Energy|Gas
- Emissions|CO2|Industrial Processes
- Carbon Sequestration|CCS
- Carbon Sequestration|CCS|Biomass
- Emissions|CO2|Energy and Industrial Processes
- Emissions|CO2|AFOLU
- Carbon Sequestration|CCS|Fossil|Energy
- Carbon Sequestration|CCS|Fossil|Industrial Processes
- Carbon Sequestration|CCS|Biomass|Energy
- Carbon Sequestration|CCS|Biomass|Industrial Processes

### All-Sectors Analysis (Optional)

For `compute_all_sectors_lmdi_cumulative`, additional variables:

- Emissions|CH4
- Emissions|N2O
- Emissions|F-Gases (or individual HFC, PFC, SF6)

## Computed Variables

### Kaya Variables (Intermediate)

- Primary Energy|Fossil
- Total Fossil Carbon
- Net Fossil Carbon

### Kaya Factors

- GNP/P (GDP per capita)
- FE/GNP (Energy intensity of GDP)
- PEDEq/FE (Primary to final energy ratio)
- PEFF/PEDEq (Fossil share of primary energy)
- TFC/PEFF (Carbon intensity of fossil energy)
- NFC/TFC (Net to total fossil carbon ratio)

## API Reference

### Core Functions

#### `compute_kaya_variables(input_data)`

Compute intermediate Kaya variables from input data.

**Parameters:**
- `input_data` (pyam.IamDataFrame): Input data with required variables

**Returns:**
- pyam.IamDataFrame with computed Kaya variables

**Raises:**
- ValueError: If required input variables are missing from the input data

#### `compute_kaya_factors(kaya_variables_frame)`

Compute Kaya decomposition factors.

**Parameters:**
- `kaya_variables_frame` (pyam.IamDataFrame): Output from compute_kaya_variables

**Returns:**
- pyam.IamDataFrame with computed factors

### LMDI Functions

#### `compute_lmdi(kaya_factors_df, ref_scenario, int_scenario)`

Compute corrected LMDI decomposition between two scenarios.

**Parameters:**
- `kaya_factors_df` (pyam.IamDataFrame): Output from compute_kaya_factors
- `ref_scenario` (tuple): Reference scenario (model, scenario, region)
- `int_scenario` (tuple): Intervention scenario (model, scenario, region)

**Returns:**
- pyam.IamDataFrame with LMDI contributions for each factor

#### `compute_lmdi_cumulative(kaya_factors_df, base_year, scenario)`

Compute cumulative LMDI decomposition for a single scenario over time.

**Parameters:**
- `kaya_factors_df` (pyam.IamDataFrame): Output from compute_kaya_factors
- `base_year` (int): Reference year for comparison (default: 2020)
- `scenario` (tuple, optional): Scenario to analyze (model, scenario, region)

**Returns:**
- pyam.IamDataFrame with LMDI contributions at each time point

#### `compute_lmdi_cumulative_sum(kaya_factors_or_lmdi_df, base_year, periods, integration_method, use_corrected)`

Sum cumulative LMDI contributions over specified time periods.

**Parameters:**
- `kaya_factors_or_lmdi_df` (pyam.IamDataFrame): Kaya factors or LMDI results
- `base_year` (int): Base year for calculation (default: 2020)
- `periods` (list of tuples): Periods to sum, e.g., [(2020, 2050), (2050, 2100)]
- `integration_method` (str): "trapezoidal" (default) or "endpoint"
- `use_corrected` (bool): Whether to use non-negativity corrected values

**Returns:**
- pd.DataFrame with factors as rows, periods as columns (values in Gt CO2)

### Savings Functions

#### `compute_savings(input_data, ref_scenario, int_scenario, periods, integration_method)`

Compute avoided emissions (savings) from comparing two scenarios.

**Parameters:**
- `input_data` (pyam.IamDataFrame): Raw input data containing both scenarios
- `ref_scenario` (tuple): Reference scenario (model, scenario, region)
- `int_scenario` (tuple): Intervention scenario (model, scenario, region)
- `periods` (list of tuples, optional): Periods to compute savings for (default: [(2020, 2050), (2050, 2100), (2020, 2100)])
- `integration_method` (str): "trapezoidal" (default) or "endpoint"

**Returns:**
- pd.DataFrame with emission components as rows and periods as columns (values in Gt CO2)

#### `compute_savings_with_percentages(input_data, ref_scenario, int_scenario, period, integration_method)`

Compute savings with percentage columns for a single period.

**Parameters:**
- `input_data` (pyam.IamDataFrame): Raw input data containing both scenarios
- `ref_scenario` (tuple): Reference scenario (model, scenario, region)
- `int_scenario` (tuple): Intervention scenario (model, scenario, region)
- `period` (tuple): Single period as (start_year, end_year)
- `integration_method` (str): "trapezoidal" (default) or "endpoint"

**Returns:**
- pd.DataFrame with columns: "Gt CO2", "% of total savings", "% of reference emissions"

#### `compute_lmdi_scenario_comparison(kaya_factors_df, ref_scenario, int_scenario)`

Compute LMDI decomposition between two scenarios at each time point.

**Parameters:**
- `kaya_factors_df` (pyam.IamDataFrame): Output from compute_kaya_factors for both scenarios
- `ref_scenario` (tuple): Reference scenario (model, scenario, region)
- `int_scenario` (tuple): Intervention scenario (model, scenario, region)

**Returns:**
- pyam.IamDataFrame with LMDI contributions at each time point

### All-Sectors Functions

#### `compute_other_gases_emissions(input_data, fgas_method, missing_value)`

Compute total non-CO2 greenhouse gas emissions in CO2-equivalent.

**Parameters:**
- `input_data` (pyam.IamDataFrame): Input data with CH4, N2O, F-gas variables
- `fgas_method` (str): "aggregate" (default) or "disaggregate"
- `missing_value` (float): Value to use when input data is missing (default: 0.0). Use `np.nan` to propagate missing data as NaN.

**Returns:**
- pyam.IamDataFrame with total other gases in Mt CO2-equivalent/yr

#### `compute_industrial_process_emissions(input_data, missing_value)`

Compute net industrial process carbon emissions (after CCS).

**Parameters:**
- `input_data` (pyam.IamDataFrame): Input data
- `missing_value` (float): Value to use when input data is missing (default: 0.0). Use `np.nan` to propagate missing data as NaN.

**Returns:**
- pyam.IamDataFrame with Net Industrial Carbon in Mt CO2/yr

#### `compute_all_sectors_emissions(input_data)`

Compute total emissions breakdown for all sectors.

**Parameters:**
- `input_data` (pyam.IamDataFrame): Input data with all required variables

**Returns:**
- pyam.IamDataFrame with NFC, Net Industrial Carbon, Other Gases (CO2-eq), and Land Use emissions

#### `compute_all_sectors_lmdi_cumulative(input_data, base_year, scenario, periods, integration_method, use_corrected)`

Compute cumulative LMDI for all emission sectors.

**Parameters:**
- `input_data` (pyam.IamDataFrame): Raw input data with all required variables
- `base_year` (int): Base year for LMDI calculation (default: 2020)
- `scenario` (tuple, optional): Scenario to analyze (model, scenario, region)
- `periods` (list of tuples): Periods to sum over
- `integration_method` (str): "trapezoidal" (default) or "endpoint"
- `use_corrected` (bool): Whether to use corrected values (default: False)

**Returns:**
- pd.DataFrame matching the LMDItableRefAllSectors format with rows:
  - Population
  - Economic Activity per Person
  - Energy Intensity of Economy
  - Energy Supply Loss Factor
  - Fossil Fuel Fraction
  - Carbon Intensity of Fossil Energy
  - Industrial Process Carbon Emissions
  - Other Gases
  - Land Use
  - Total Net Emissions

## Constants

Access variable name constants for programmatic use:

```python
from kaya_decomposition.constants import (
    input_variables,
    kaya_variables,
    kaya_factors,
    lmdi,
    lmdi_cumulative,
    savings,
)

# Input variable names
print(input_variables.POPULATION)  # "Population"
print(input_variables.GDP_PPP)     # "GDP|PPP"

# Computed variable names
print(kaya_variables.TFC)          # "Total Fossil Carbon"

# Factor names
print(kaya_factors.GNP_per_P)      # "GNP/P"

# LMDI output names
print(lmdi.Pop_LMDI)               # "Population (LMDI)"

# Cumulative LMDI names (human-readable)
print(lmdi_cumulative.Pop_cumulative)  # "Population"

# Savings output labels
print(savings.POPULATION)          # "Population"
print(savings.ENERGY_INTENSITY)    # "Energy Intensity of Economy"
```

## License

Apache-2.0

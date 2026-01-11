# kaya-decomposition

Kaya decomposition analysis for integrated-assessment scenario data.

## Overview

This library provides tools for computing Kaya decomposition factors from IAMC-format scenario data. The Kaya identity decomposes CO2 emissions into contributing factors: population, GDP per capita, energy intensity, and carbon intensity.

## Installation

```bash
pip install kaya-decomposition
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

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

## Required Input Variables

The following variables must be present in your input data:

- Population
- GDP|PPP or GDP|MER
- Final Energy
- Primary Energy
- Primary Energy|Coal
- Primary Energy|Oil
- Primary Energy|Gas
- Emissions|CO2|Industrial Processes
- Emissions|CO2|Carbon Capture and Storage
- Emissions|CO2|Carbon Capture and Storage|Biomass
- Emissions|CO2|Fossil Fuels and Industry
- Emissions|CO2|AFOLU
- Carbon Sequestration|CCS|Fossil|Energy
- Carbon Sequestration|CCS|Fossil|Industrial Processes
- Carbon Sequestration|CCS|Biomass|Energy
- Carbon Sequestration|CCS|Biomass|Industrial Processes

## Computed Variables

### Kaya Variables (intermediate)

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

### `compute_kaya_variables(input_data)`

Compute intermediate Kaya variables from input data.

**Parameters:**
- `input_data` (pyam.IamDataFrame): Input data with required variables

**Returns:**
- pyam.IamDataFrame with computed variables, or None if input incomplete

### `compute_kaya_factors(kaya_variables_frame)`

Compute Kaya decomposition factors.

**Parameters:**
- `kaya_variables_frame` (pyam.IamDataFrame): Output from compute_kaya_variables

**Returns:**
- pyam.IamDataFrame with computed factors

## Constants

Access variable name constants for programmatic use:

```python
from kaya_decomposition.constants import input_variables, kaya_variables, kaya_factors

# Input variable names
print(input_variables.POPULATION)  # "Population"
print(input_variables.GDP_PPP)     # "GDP|PPP"

# Computed variable names
print(kaya_variables.TFC)          # "Total Fossil Carbon"

# Factor names
print(kaya_factors.GNP_per_P)      # "GNP/P"
```

## License

Apache-2.0

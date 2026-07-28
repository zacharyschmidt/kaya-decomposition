# Kaya Decomposition Library Documentation

A Python library for computing Kaya decomposition factors from IAMC-format integrated assessment scenario data.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [The Kaya Identity](#the-kaya-identity)
5. [Required Input Variables](#required-input-variables)
6. [API Reference](#api-reference)
   - [Core Functions](#core-functions)
   - [LMDI Decomposition](#lmdi-decomposition)
   - [All-Sectors Analysis](#all-sectors-analysis)
7. [Usage Examples](#usage-examples)
8. [Constants Reference](#constants-reference)
9. [Comparison to Excel Implementation](#comparison-to-excel-implementation)

---

## Overview

The `kaya-decomposition` library provides tools for:

- Computing intermediate Kaya variables from scenario data
- Calculating Kaya decomposition factors (ratios)
- Performing LMDI (Logarithmic Mean Divisia Index) decomposition to attribute emissions changes
- Analyzing all emissions sectors including non-CO2 gases

The library is designed to work with [pyam](https://pyam-iamc.readthedocs.io/), a Python package for analysis of integrated assessment scenarios using the IAMC data format.

---

## Installation

### From PyPI (when published)

```bash
pip install kaya-decomposition
```

### For Development

```bash
git clone <repository-url>
cd kaya-decomposition
pip install -e ".[dev]"
```

### Dependencies

- Python 3.8+
- pyam-iamc
- numpy
- pandas

---

## Quick Start

```python
import pyam
from kaya_decomposition import (
    compute_kaya_variables,
    compute_kaya_factors,
    compute_lmdi_cumulative,
    compute_lmdi_cumulative_sum,
)

# Load your IAMC-format data
df = pyam.IamDataFrame("your_scenario_data.csv")

# Step 1: Compute intermediate Kaya variables
kaya_vars = compute_kaya_variables(df)

# Step 2: Compute Kaya decomposition factors (ratios)
kaya_factors = compute_kaya_factors(kaya_vars)

# Step 3: Compute LMDI decomposition relative to base year
lmdi_results = compute_lmdi_cumulative(kaya_factors, base_year=2020)

# Step 4: Sum LMDI contributions over time periods
period_sums = compute_lmdi_cumulative_sum(lmdi_results)
print(period_sums)
```

---

## The Kaya Identity

The [Kaya Identity](https://en.wikipedia.org/wiki/Kaya_identity) relates anthropogenic CO2 emissions to key socioeconomic factors:

$$
\text{CO}_2 = P \times \frac{\text{GDP}}{P} \times \frac{E}{{\text{GDP}}} \times \frac{\text{CO}_2}{E}
$$

Where:
- **P** = Population
- **GDP/P** = GDP per capita (economic output per person)
- **E/GDP** = Energy intensity of GDP (energy per unit economic output)
- **CO₂/E** = Carbon intensity of energy

This library extends the basic Kaya identity with a more detailed decomposition:

| Factor | Symbol | Description |
|--------|--------|-------------|
| Population | P | Total population |
| GDP per capita | GNP/P | Economic output per person |
| Energy intensity | FE/GNP | Final energy per unit GDP |
| Energy conversion | PEDEq/FE | Primary energy to final energy ratio |
| Fossil share | PEFF/PEDEq | Fossil fraction of primary energy |
| Carbon intensity | TFC/PEFF | Carbon per unit fossil energy |
| Net/Total ratio | NFC/TFC | Net fossil carbon to total fossil carbon |

### Key Variables

- **TFC (Total Fossil Carbon)**: Total CO2 emissions from fossil fuel combustion before accounting for CCS
- **NFC (Net Fossil Carbon)**: Net CO2 emissions after carbon capture and storage

---

## Required Input Variables

Your input `IamDataFrame` must contain the following variables:

| Variable Name | Description | Typical Unit |
|---------------|-------------|--------------|
| `Population` | Total population | million |
| `GDP\|PPP` or `GDP\|MER` | Gross domestic product | billion USD/yr |
| `Final Energy` | Total final energy consumption | EJ/yr |
| `Primary Energy` | Total primary energy supply | EJ/yr |
| `Primary Energy\|Coal` | Primary energy from coal | EJ/yr |
| `Primary Energy\|Oil` | Primary energy from oil | EJ/yr |
| `Primary Energy\|Gas` | Primary energy from natural gas | EJ/yr |
| `Emissions\|CO2\|Fossil Fuels and Industry` | CO2 from fossil fuels and industry | Mt CO2/yr |
| `Emissions\|CO2\|Industrial Processes` | CO2 from industrial processes | Mt CO2/yr |
| `Emissions\|CO2\|AFOLU` | CO2 from land use | Mt CO2/yr |
| `Emissions\|CO2\|Carbon Capture and Storage` | CO2 captured (total) | Mt CO2/yr |
| `Emissions\|CO2\|Carbon Capture and Storage\|Biomass` | CO2 captured from biomass | Mt CO2/yr |
| `Carbon Sequestration\|CCS\|Fossil\|Energy` | CCS from fossil energy | Mt CO2/yr |
| `Carbon Sequestration\|CCS\|Fossil\|Industrial Processes` | CCS from fossil industrial | Mt CO2/yr |
| `Carbon Sequestration\|CCS\|Biomass\|Energy` | CCS from biomass energy | Mt CO2/yr |
| `Carbon Sequestration\|CCS\|Biomass\|Industrial Processes` | CCS from biomass industrial | Mt CO2/yr |

### Optional Variables (for All-Sectors Analysis)

| Variable Name | Description | Typical Unit |
|---------------|-------------|--------------|
| `Emissions\|CH4` | Methane emissions | Mt CH4/yr |
| `Emissions\|N2O` | Nitrous oxide emissions | kt N2O/yr |
| `Emissions\|F-Gases` | Fluorinated gases (CO2-equivalent) | Mt CO2-equiv/yr |
| `Carbon Removal\|Geological Storage\|Biomass` | BECCS, modern IAMC naming (preferred over legacy `Carbon Sequestration\|CCS\|Biomass\|...`) | Mt CO2/yr |
| `Carbon Removal\|Geological Storage\|Direct Air Capture` | DACCS | Mt CO2/yr |
| `Carbon Removal\|Land Use` | Land-based removal (afforestation, biochar, soil carbon) | Mt CO2/yr |

---

## API Reference

### Core Functions

#### `compute_kaya_variables(input_data)`

Compute intermediate Kaya variables from raw input data.

**Parameters:**
- `input_data` (`pyam.IamDataFrame`): Input data containing required variables

**Returns:**
- `pyam.IamDataFrame` with computed variables

**Raises:**
- `ValueError` if required input variables are missing

**Computed Variables:**
- `Primary Energy|Fossil`: Sum of coal, oil, and gas primary energy
- `Total Fossil Carbon (TFC)`: Gross fossil CO2 before CCS
- `Net Fossil Carbon (NFC)`: Net fossil CO2 after CCS

**Example:**
```python
import pyam
from kaya_decomposition import compute_kaya_variables

df = pyam.IamDataFrame("data.csv")
kaya_vars = compute_kaya_variables(df)

# View computed variables
print(kaya_vars.variable)
# ['Population', 'GDP|PPP', 'Final Energy', 'Primary Energy', 
#  'Primary Energy|Fossil', 'Total Fossil Carbon', 'Net Fossil Carbon']
```

---

#### `compute_kaya_factors(kaya_variables_frame)`

Compute Kaya decomposition factors (ratios) from intermediate variables.

**Parameters:**
- `kaya_variables_frame` (`pyam.IamDataFrame`): Output from `compute_kaya_variables()`

**Returns:**
- `pyam.IamDataFrame` with computed Kaya factors

**Raises:**
- `ValueError` if input is None, not an IamDataFrame, or empty

**Computed Factors:**

| Factor | Formula | Meaning |
|--------|---------|---------|
| `GNP/P` | GDP ÷ Population | GDP per capita |
| `FE/GNP` | Final Energy ÷ GDP | Energy intensity of economy |
| `PEDEq/FE` | Primary Energy ÷ Final Energy | Energy conversion losses |
| `PEFF/PEDEq` | Fossil PE ÷ Primary Energy | Fossil share of energy mix |
| `TFC/PEFF` | Total Fossil Carbon ÷ Fossil PE | Carbon per unit fossil energy |
| `NFC/TFC` | Net Fossil Carbon ÷ TFC | Effect of CCS |

**Example:**
```python
from kaya_decomposition import compute_kaya_variables, compute_kaya_factors

kaya_vars = compute_kaya_variables(df)
factors = compute_kaya_factors(kaya_vars)

# Access a specific factor for analysis
gnp_per_p = factors.filter(variable="GNP/P")
print(gnp_per_p.data)
```

---

### LMDI Decomposition

The library supports two types of LMDI decomposition:

1. **Scenario Comparison** (`compute_lmdi`): Compare two scenarios at the same time points
2. **Cumulative Over Time** (`compute_lmdi_cumulative`): Track changes within a single scenario over time

#### `compute_lmdi(kaya_factors_df, ref_scenario, int_scenario)`

Compute LMDI decomposition between a reference and intervention scenario.

**Parameters:**
- `kaya_factors_df` (`pyam.IamDataFrame`): Output from `compute_kaya_factors()` containing both scenarios
- `ref_scenario` (`tuple`): Reference scenario as `(model, scenario, region)`
- `int_scenario` (`tuple`): Intervention scenario as `(model, scenario, region)`

**Returns:**
- `pyam.IamDataFrame` with LMDI contributions for each factor

**Example:**
```python
from kaya_decomposition import compute_kaya_variables, compute_kaya_factors, compute_lmdi

# Load data with two scenarios
df = pyam.IamDataFrame("scenarios.csv")

kaya_vars = compute_kaya_variables(df)
factors = compute_kaya_factors(kaya_vars)

# Compare baseline to mitigation scenario
lmdi = compute_lmdi(
    factors,
    ref_scenario=("MODEL", "Baseline", "World"),
    int_scenario=("MODEL", "1.5C", "World"),
)

# Each LMDI term shows how much that factor contributed to the emissions difference
print(lmdi.filter(variable="Population (LMDI)").data)
```

---

#### `compute_lmdi_cumulative(kaya_factors_df, base_year=2020, scenario=None)`

Compute cumulative LMDI decomposition for a single scenario over time.

**Parameters:**
- `kaya_factors_df` (`pyam.IamDataFrame`): Output from `compute_kaya_factors()`
- `base_year` (`int`): Reference year for comparison (default: 2020)
- `scenario` (`tuple`, optional): Scenario as `(model, scenario, region)`. If None, uses first available.

**Returns:**
- `pyam.IamDataFrame` with LMDI contributions at each time point

**Notes:**
- Uses the LMDI-I additive formula
- Contributions sum to the actual TFC change from base year
- A correction is applied to ensure non-negative contributions

**Example:**
```python
from kaya_decomposition import (
    compute_kaya_variables, 
    compute_kaya_factors, 
    compute_lmdi_cumulative
)

kaya_vars = compute_kaya_variables(df)
factors = compute_kaya_factors(kaya_vars)

# Calculate LMDI relative to 2020
lmdi = compute_lmdi_cumulative(
    factors, 
    base_year=2020,
    scenario=("IMAGE 3.0.1", "SSP2-Baseline", "World")
)

# View population contribution over time
pop_contrib = lmdi.filter(variable="Population")
print(pop_contrib.data)
```

---

#### `compute_lmdi_cumulative_sum(lmdi_cumulative_df, periods=None)`

Sum LMDI contributions over specified time periods.

**Parameters:**
- `lmdi_cumulative_df` (`pyam.IamDataFrame`): Output from `compute_lmdi_cumulative()`
- `periods` (`list[tuple]`, optional): List of `(start_year, end_year)` periods. Default: `[(2020, 2050), (2050, 2100), (2020, 2100)]`

**Returns:**
- `pd.DataFrame` with factors as rows and periods as columns

**Example:**
```python
from kaya_decomposition import compute_lmdi_cumulative_sum

# Sum contributions over standard periods
period_table = compute_lmdi_cumulative_sum(lmdi)
print(period_table)
```

**Output:**
```
                                    2020 to 2050  2050 to 2100  2020 to 2100
Population                              1234.56       2345.67       3580.23
Economic Activity per Person            5678.90       8901.23      14580.13
Energy Intensity of Economy            -2345.67      -4567.89      -6913.56
...
```

---

### All-Sectors Analysis

These functions extend the analysis beyond fossil CO2 to include all greenhouse gas sectors.

#### `compute_other_gases_emissions(input_data, fgas_method="aggregate")`

Compute non-CO2 greenhouse gas emissions in CO2-equivalent.

**Parameters:**
- `input_data` (`pyam.IamDataFrame`): Input data with CH4, N2O, and F-gas variables
- `fgas_method` (`str`): Either `"aggregate"` (use pre-aggregated F-gases) or `"disaggregate"` (compute from HFC, PFC, SF6)

**Returns:**
- `pyam.IamDataFrame` with variable `Emissions|Other Gases|CO2-equivalent`

**GWP Values Used (IPCC AR6):**
- CH4: 27.9
- N2O: 273
- HFC134a: 1530
- CF4 (PFC): 7380
- SF6: 25200

---

#### `compute_industrial_process_emissions(input_data)`

Compute net industrial process carbon emissions.

**Returns:**
- `pyam.IamDataFrame` with variable `Net Industrial Carbon`

---

#### `compute_total_cdr(input_data)`

Compute total carbon dioxide removal (CDR).

**Scope:** CDR includes only technologies that remove CO2 already in the
atmosphere or biosphere — bioenergy with carbon capture and storage
(BECCS), direct air capture (DACCS), and land-based removal. It
excludes fossil fuel carbon capture, which prevents a new emission
(abatement) rather than removing existing CO2, and so is not CDR.

For the biomass (BECCS) component, the modern IAMC `Carbon
Removal|Geological Storage|Biomass` variable is preferred when
reported; otherwise this falls back to the legacy `Carbon
Sequestration|CCS|Biomass|Energy` + `...|Industrial Processes` split
used by current datasets. The two are never summed together. DACCS
and land-based removal have no legacy equivalent and default to zero
when absent.

**Returns:**
- `pyam.IamDataFrame` with variable `Carbon Dioxide Removal`, reported
  as a negative value (Mt CO2/yr)

---

#### `compute_all_sectors_lmdi_cumulative(input_data, base_year=2020, scenario=None, periods=None)`

Compute complete LMDI analysis for all emission sectors.

**Parameters:**
- `input_data` (`pyam.IamDataFrame`): Raw input data
- `base_year` (`int`): Reference year (default: 2020)
- `scenario` (`tuple`, optional): Scenario identifiers
- `periods` (`list[tuple]`, optional): Time periods to sum over

**Returns:**
- `pd.DataFrame` with all sectors including:
  - Kaya factors (Population, GDP/capita, Energy intensity, etc.)
  - Industrial Process Carbon Emissions
  - Other Gases (CH4, N2O, F-gases)
  - Land Use (AFOLU)
  - Total CDR (carbon dioxide removal)
  - Total Net Emissions

**Example:**
```python
from kaya_decomposition import compute_all_sectors_lmdi_cumulative

result = compute_all_sectors_lmdi_cumulative(
    df,
    base_year=2020,
    scenario=("IMAGE 3.0.1", "SSP2-Baseline", "World"),
    periods=[(2020, 2050), (2050, 2100), (2020, 2100)]
)

print(result)
```

**Output:**
```
                                           2020 to 2050  2050 to 2100  2020 to 2100
Population                                     1234.56       2345.67       3580.23
Economic Activity per Person                   5678.90       8901.23      14580.13
Energy Intensity of Economy                   -2345.67      -4567.89      -6913.56
Energy Supply Loss Factor                       123.45        234.56        358.01
Fossil Fuel Fraction                            567.89        890.12       1458.01
Carbon Intensity of Fossil Energy              -345.67       -567.89       -913.56
Industrial Process Carbon Emissions             234.56        456.78        691.34
Other Gases                                    1234.56       1567.89       2802.45
Land Use                                        345.67       -234.56        111.11
Total CDR                                      -456.78       -890.12      -1346.90
Total Net Emissions                            6271.47      12136.79      18408.26
```

---

## Usage Examples

### Example 1: Basic Kaya Factor Analysis

```python
import pyam
from kaya_decomposition import compute_kaya_variables, compute_kaya_factors

# Load data
df = pyam.IamDataFrame("scenario_data.csv")

# Compute Kaya analysis (raises ValueError if required variables are missing)
kaya_vars = compute_kaya_variables(df)
factors = compute_kaya_factors(kaya_vars)

# Plot energy intensity over time
fe_gnp = factors.filter(variable="FE/GNP")
fe_gnp.plot()
```

### Example 2: Comparing Two Scenarios

```python
import pyam
from kaya_decomposition import (
    compute_kaya_variables,
    compute_kaya_factors, 
    compute_lmdi
)

# Load data with baseline and mitigation scenarios
df = pyam.IamDataFrame("multi_scenario.csv")

# Process both scenarios together
kaya_vars = compute_kaya_variables(df)
factors = compute_kaya_factors(kaya_vars)

# Attribute emissions differences to factors
lmdi = compute_lmdi(
    factors,
    ref_scenario=("MyModel", "Baseline", "World"),
    int_scenario=("MyModel", "NetZero2050", "World")
)

# Analyze what drove the emissions reduction
for var in lmdi.variable:
    data = lmdi.filter(variable=var, year=2050).data
    print(f"{var}: {data['value'].values[0]:.1f} Mt CO2")
```

### Example 3: Historical Driver Analysis

```python
import pyam
from kaya_decomposition import (
    compute_kaya_variables,
    compute_kaya_factors,
    compute_lmdi_cumulative,
    compute_lmdi_cumulative_sum,
)

# Load historical data
df = pyam.IamDataFrame("historical_data.csv")

# Full analysis pipeline
kaya_vars = compute_kaya_variables(df)
factors = compute_kaya_factors(kaya_vars)
lmdi = compute_lmdi_cumulative(factors, base_year=1990)

# Custom periods
period_table = compute_lmdi_cumulative_sum(
    lmdi,
    periods=[(1990, 2000), (2000, 2010), (2010, 2020)]
)

print("Emissions drivers by decade:")
print(period_table)
```

### Example 4: Using Constants for Programmatic Access

```python
from kaya_decomposition import (
    input_variables,
    kaya_variables,
    kaya_factors as kf,
    lmdi_cumulative,
)

# Check if required variables exist in your data
required = input_variables.REQUIRED_VARIABLES
missing = [v for v in required if v not in df.variable]
if missing:
    print(f"Missing: {missing}")

# Access specific computed variables by name
tfc_data = factors.filter(variable=kaya_variables.TFC)

# Filter LMDI results by factor name
energy_intensity = lmdi.filter(variable=lmdi_cumulative.FE_per_GNP_cumulative)
```

---

## Constants Reference

The library exports constant modules for programmatic access to variable names:

### `input_variables`

```python
from kaya_decomposition import input_variables

input_variables.POPULATION           # "Population"
input_variables.GDP_PPP              # "GDP|PPP"
input_variables.GDP_MER              # "GDP|MER"
input_variables.FINAL_ENERGY         # "Final Energy"
input_variables.PRIMARY_ENERGY       # "Primary Energy"
input_variables.PRIMARY_ENERGY_COAL  # "Primary Energy|Coal"
input_variables.PRIMARY_ENERGY_OIL   # "Primary Energy|Oil"
input_variables.PRIMARY_ENERGY_GAS   # "Primary Energy|Gas"
# ... and more

# GWP values for non-CO2 gases
input_variables.GWP_CH4              # 27.9
input_variables.GWP_N2O              # 273
```

### `kaya_variables`

```python
from kaya_decomposition import kaya_variables

kaya_variables.PRIMARY_ENERGY_FF     # "Primary Energy|Fossil"
kaya_variables.TFC                   # "Total Fossil Carbon"
kaya_variables.NFC                   # "Net Fossil Carbon"
```

### `kaya_factors`

```python
from kaya_decomposition import kaya_factors

kaya_factors.GNP_per_P               # "GNP/P"
kaya_factors.FE_per_GNP              # "FE/GNP"
kaya_factors.PEdeq_per_FE            # "PEDEq/FE"
kaya_factors.PEFF_per_PEDEq          # "PEFF/PEDEq"
kaya_factors.TFC_per_PEFF            # "TFC/PEFF"
kaya_factors.NFC_per_TFC             # "NFC/TFC"
```

### `lmdi` (Scenario Comparison)

```python
from kaya_decomposition import lmdi

lmdi.Pop_LMDI                        # "Population (LMDI)"
lmdi.GNP_per_P_LMDI                  # "GNP/P (LMDI)"
lmdi.FE_per_GNP_LMDI                 # "FE/GNP (LMDI)"
# ... etc
```

### `lmdi_cumulative` (Over-Time Analysis)

```python
from kaya_decomposition import lmdi_cumulative

lmdi_cumulative.Pop_cumulative           # "Population"
lmdi_cumulative.GNP_per_P_cumulative     # "Economic Activity per Person"
lmdi_cumulative.FE_per_GNP_cumulative    # "Energy Intensity of Economy"
lmdi_cumulative.PEdeq_per_FE_cumulative  # "Energy Supply Loss Factor"
lmdi_cumulative.PEFF_per_PEDEq_cumulative # "Fossil Fuel Fraction"
lmdi_cumulative.TFC_per_PEFF_cumulative  # "Carbon Intensity of Fossil Energy"
lmdi_cumulative.Industrial_Process       # "Industrial Process Carbon Emissions"
lmdi_cumulative.Other_Gases              # "Other Gases"
lmdi_cumulative.Land_Use                 # "Land Use"
lmdi_cumulative.Total_CDR                # "Total CDR"
lmdi_cumulative.Total_Net_Emissions      # "Total Net Emissions"
```

---

## Comparison to Excel Implementation

This library is designed to replicate and extend the analysis from the Excel workbook `vanVuurenIMAGE_15_TOT_19_TFC_currentcopy.xlsm`.

### Excel Workbook Data Flow

The original Excel implementation flows through these sheets:

| Sheet | Description | Python Equivalent |
|-------|-------------|-------------------|
| `Ref Data input` | Raw scenario data | Input `IamDataFrame` |
| `Ref Data` | Extracted IAMC variables | Input `IamDataFrame` |
| `FossilEneEmissionsAccountingRef` | TFC/NFC calculation | `compute_kaya_variables()` |
| `PEfossilRef` | Fossil energy calculation | `compute_kaya_variables()` |
| `ExpKayaFactorsRef` | P, GNP, FE, PEDEq, PEFF, TFC, NFC | `compute_kaya_factors()` |
| `ExpKayaRatiosRef` | Kaya ratios | `compute_kaya_factors()` |
| `LMDI 1 MethodRefCumulative` | LMDI vs base year | `compute_lmdi_cumulative()` |
| `LMDItableRefAllSectors` | Period sums | `compute_all_sectors_lmdi_cumulative()` |

### Key Differences

| Aspect | Excel Implementation | Python Library |
|--------|---------------------|----------------|
| **Data format** | Hardcoded cells | IAMC-format IamDataFrame |
| **Flexibility** | Single scenario | Any number of scenarios |
| **Regions** | Single region | Multiple regions supported |
| **Extensibility** | Manual formula updates | Modular functions |
| **Validation** | Visual inspection | Unit tests with expected values |
| **Integration** | Standalone | Works with pyam ecosystem |

### LMDI Calculation Method

The library implements the **LMDI-I additive decomposition** with the same non-negativity correction as the Excel:

**LMDI-I Formula:**
$$
\Delta V_i = L(V_t, V_0) \times \ln\left(\frac{x_{i,t}}{x_{i,0}}\right)
$$

Where:
- $L(a, b) = \frac{a - b}{\ln a - \ln b}$ (logarithmic mean)
- $V$ = aggregate variable (TFC)
- $x_i$ = factor i

**Non-Negativity Correction:**

1. Calculate uncorrected LMDI for each factor
2. Clip negative values to zero → `non_neg`
3. Sum: `total_non_neg = sum(non_neg)`
4. Calculate actual difference: `actual_diff = TFC_t - TFC_0`
5. Calculate residual: `residual = total_non_neg - actual_diff`
6. Distribute: `percent_i = non_neg_i / total_non_neg`
7. Correct: `corrected_i = non_neg_i - (percent_i × residual)`

This ensures corrected contributions sum exactly to the actual change.

### Validation Results

The Python library has been validated against the Excel file with the IMAGE 3.0.1 SSP2-Baseline scenario. Key comparisons:

| Metric | Excel Value | Python Value | Match |
|--------|-------------|--------------|-------|
| TFC (2020) | 36419.17 | 36419.17 | ✓ |
| TFC (2050) | 48651.53 | 48651.53 | ✓ |
| TFC (2100) | 69304.62 | 69304.62 | ✓ |
| Population LMDI (2020-2050) | ~1500 | ~1500 | ✓ |
| Energy Intensity LMDI (2020-2050) | ~-3500 | ~-3500 | ✓ |

### Running Validation Tests

```bash
# Run all tests
pytest tests/

# Run Excel validation tests specifically
pytest tests/test_excel_validation.py -v
```

---

## Troubleshooting

### `ValueError: missing required input variables`

This error from `compute_kaya_variables()` means required variables are missing. The error message lists the missing variables. You can also check manually:

```python
from kaya_decomposition import input_variables

required = input_variables.REQUIRED_VARIABLES
present = df.variable
missing = [v for v in required if v not in present]
print(f"Missing variables: {missing}")
```

### Unit Mismatches

The library uses `ignore_units=True` for many operations because different data sources use different unit conventions. Ensure your data has consistent units for meaningful comparisons.

### Empty Results

If filtering returns empty results, check your model/scenario/region identifiers:

```python
print(df.model)     # Available models
print(df.scenario)  # Available scenarios  
print(df.region)    # Available regions
```

---

## License

Apache-2.0

---

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest tests/`
2. Code follows existing style
3. New features include tests
4. Documentation is updated

---

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{kaya_decomposition,
  title = {kaya-decomposition: Kaya decomposition analysis for integrated-assessment scenario data},
  author = {IIASA},
  year = {2024},
  url = {https://github.com/iiasa/kaya-decomposition}
}
```

"""Constants defining input variable names for Kaya analysis."""

# Required input variables for core Kaya decomposition
POPULATION = "Population"
GDP_MER = "GDP|MER"
GDP_PPP = "GDP|PPP"
FINAL_ENERGY = "Final Energy"
PRIMARY_ENERGY = "Primary Energy"
PRIMARY_ENERGY_COAL = "Primary Energy|Coal"
PRIMARY_ENERGY_OIL = "Primary Energy|Oil"
PRIMARY_ENERGY_GAS = "Primary Energy|Gas"
EMISSIONS_CO2_INDUSTRIAL_PROCESSES = "Emissions|CO2|Industrial Processes"
CCS = "Carbon Sequestration|CCS"
CCS_BIOMASS = "Carbon Sequestration|CCS|Biomass"
EMISSIONS_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES = "Emissions|CO2|Energy and Industrial Processes"
EMISSIONS_CO2_AFOLU = "Emissions|CO2|AFOLU"
CCS_FOSSIL_ENERGY = "Carbon Sequestration|CCS|Fossil|Energy"
CCS_FOSSIL_INDUSTRY = "Carbon Sequestration|CCS|Fossil|Industrial Processes"
CCS_BIOMASS_ENERGY = "Carbon Sequestration|CCS|Biomass|Energy"
CCS_BIOMASS_INDUSTRY = "Carbon Sequestration|CCS|Biomass|Industrial Processes"

# List of required variables for core Kaya decomposition
# (used by compute_kaya_variables to check for complete data)
REQUIRED_VARIABLES = [
    POPULATION,
    GDP_MER,
    GDP_PPP,
    FINAL_ENERGY,
    PRIMARY_ENERGY,
    PRIMARY_ENERGY_COAL,
    PRIMARY_ENERGY_OIL,
    PRIMARY_ENERGY_GAS,
    EMISSIONS_CO2_INDUSTRIAL_PROCESSES,
    CCS,
    CCS_BIOMASS,
    EMISSIONS_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES,
    EMISSIONS_CO2_AFOLU,
    CCS_FOSSIL_ENERGY,
    CCS_FOSSIL_INDUSTRY,
    CCS_BIOMASS_ENERGY,
    CCS_BIOMASS_INDUSTRY,
]

# Optional input variables for all-sectors analysis
# Non-CO2 greenhouse gases (for OtherGases calculation)
EMISSIONS_CH4 = "Emissions|CH4"
EMISSIONS_N2O = "Emissions|N2O"
EMISSIONS_FGASES = "Emissions|F-Gases"
EMISSIONS_HFC = "Emissions|HFC"
EMISSIONS_PFC = "Emissions|PFC"
EMISSIONS_SF6 = "Emissions|SF6"

# GWP conversion factors (AR6 100-year GWP values from Excel)
GWP_CH4 = 27.9  # CO2-equivalent per Mt CH4
GWP_N2O = 273   # CO2-equivalent per kt N2O
GWP_HFC134A = 1530  # for HFC134a-equivalent
GWP_CF4 = 7380  # for CF4-equivalent (PFC)
GWP_SF6 = 25200  # for SF6

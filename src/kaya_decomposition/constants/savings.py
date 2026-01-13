"""Constants for savings calculations.

These labels match the Excel Savings tab output format.
"""

# Row labels for Savings table
REF_CUMULATIVE = "Reference case cumulative emissions"
INT_CUMULATIVE = "Intervention case cumulative emissions"
DIFFERENCE = "Difference"

# Kaya factor labels (matching lmdi_cumulative naming)
POPULATION = "Population"
ECONOMIC_ACTIVITY = "Economic Activity per Person"
ENERGY_INTENSITY = "Energy Intensity of Economy"
ENERGY_SUPPLY_LOSS = "Energy Supply Loss Factor"
FOSSIL_FRACTION = "Fossil Fuel Fraction"
CARBON_INTENSITY = "Carbon Intensity of Fossil Energy"

# Non-Kaya sectors
INDUSTRIAL_PROCESS = "Industrial Process Carbon"
LAND_USE = "Land Use"
OTHER_GASES = "Other Gases"

# CCS contributions from Intervention
FOSSIL_CCS = "Fossil CCS (Intervention)"
BIOMASS_CCS = "Biomass CCS (Intervention)"

# Total
TOTAL_NET = "Total/Net"

# Column labels
ABS_VALUE = "Gt CO2"
PCT_OF_TOTAL = "% of total savings"
PCT_OF_REF = "% of reference emissions"

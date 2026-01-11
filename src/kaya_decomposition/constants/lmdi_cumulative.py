"""Constants defining cumulative LMDI output variable names.

These are human-readable labels matching the Excel LMDItableRefAllSectors format.
"""

# Kaya factor contributions (cumulative LMDI)
Pop_cumulative = "Population"
GNP_per_P_cumulative = "Economic Activity per Person"
FE_per_GNP_cumulative = "Energy Intensity of Economy"
PEdeq_per_FE_cumulative = "Energy Supply Loss Factor"
PEFF_per_PEDEq_cumulative = "Fossil Fuel Fraction"
TFC_per_PEFF_cumulative = "Carbon Intensity of Fossil Energy"

# Additional sectors (outside core Kaya identity)
Industrial_Process = "Industrial Process Carbon Emissions"
Other_Gases = "Other Gases"
Land_Use = "Land Use"
Total_Net_Emissions = "Total Net Emissions"

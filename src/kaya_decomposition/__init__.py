"""Kaya decomposition analysis for integrated-assessment scenario data."""

from kaya_decomposition.variables import compute_kaya_variables
from kaya_decomposition.factors import compute_kaya_factors
from kaya_decomposition.lmdi import compute_lmdi
from kaya_decomposition.lmdi_cumulative import (
    compute_lmdi_cumulative,
    compute_lmdi_cumulative_sum,
)
from kaya_decomposition.all_sectors import (
    compute_other_gases_emissions,
    compute_industrial_process_emissions,
    compute_all_sectors_emissions,
    compute_all_sectors_lmdi_cumulative,
)
from kaya_decomposition.savings import (
    compute_savings,
    compute_savings_with_percentages,
    compute_lmdi_scenario_comparison,
)
from kaya_decomposition.constants import (
    input_variables,
    kaya_variables,
    kaya_factors,
    lmdi,
    lmdi_cumulative,
    savings,
)

__version__ = "0.4.0"

__all__ = [
    # Core Kaya analysis
    "compute_kaya_variables",
    "compute_kaya_factors",
    # Scenario comparison LMDI
    "compute_lmdi",
    # Cumulative LMDI (single scenario over time)
    "compute_lmdi_cumulative",
    "compute_lmdi_cumulative_sum",
    # All sectors analysis
    "compute_other_gases_emissions",
    "compute_industrial_process_emissions",
    "compute_all_sectors_emissions",
    "compute_all_sectors_lmdi_cumulative",
    # Savings (two-scenario comparison over time)
    "compute_savings",
    "compute_savings_with_percentages",
    "compute_lmdi_scenario_comparison",
    # Constants
    "input_variables",
    "kaya_variables",
    "kaya_factors",
    "lmdi",
    "lmdi_cumulative",
    "savings",
]

"""Kaya decomposition analysis for integrated-assessment scenario data."""

from kaya_decomposition.variables import compute_kaya_variables
from kaya_decomposition.factors import compute_kaya_factors
from kaya_decomposition.lmdi import compute_lmdi
from kaya_decomposition.constants import (
    input_variables,
    kaya_variables,
    kaya_factors,
    lmdi,
)

__version__ = "0.1.0"

__all__ = [
    "compute_kaya_variables",
    "compute_kaya_factors",
    "compute_lmdi",
    "input_variables",
    "kaya_variables",
    "kaya_factors",
    "lmdi",
]

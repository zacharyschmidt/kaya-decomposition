"""Shared utility functions for Kaya decomposition analysis."""


def trapezoidal_integrate(var_data, start_year, end_year):
    """Integrate values using trapezoidal rule.

    Parameters
    ----------
    var_data : pd.DataFrame
        Data for a single variable with 'year' and 'value' columns.
    start_year : int
        Start of period (exclusive - base year has 0 contribution).
    end_year : int
        End of period (inclusive).

    Returns
    -------
    float
        Integrated sum in original units.

    Notes
    -----
    The trapezoidal rule approximates the integral by summing the areas
    of trapezoids formed between consecutive data points. For segments
    that only partially overlap with the integration period, linear
    interpolation is used to estimate values at the period boundaries.
    """
    years = sorted(var_data["year"].unique())

    total = 0
    prev_year = None
    prev_val = None

    for year in years:
        val = var_data[var_data["year"] == year]["value"].values[0]

        if prev_year is not None:
            # Only integrate segments within the period
            seg_start = max(prev_year, start_year)
            seg_end = min(year, end_year)

            if seg_start < seg_end:
                # Interpolate values at segment boundaries if needed
                if prev_year < seg_start:
                    # Linear interpolation to get value at seg_start
                    frac = (seg_start - prev_year) / (year - prev_year)
                    v1 = prev_val + frac * (val - prev_val)
                else:
                    v1 = prev_val

                if year > seg_end:
                    # Linear interpolation to get value at seg_end
                    frac = (seg_end - prev_year) / (year - prev_year)
                    v2 = prev_val + frac * (val - prev_val)
                else:
                    v2 = val

                # Trapezoidal area
                area = (v1 + v2) / 2 * (seg_end - seg_start)
                total += area

        prev_year = year
        prev_val = val

    return total

import pandas as pd

from mlutils.timeseries import extract_true_values_for_forecasts


__all__ = ["calculate_residuals"]


def calculate_residuals(forecasts, true_values):
    true_values_indices, true_values = extract_true_values_for_forecasts(true_values, forecasts)

    residuals = forecasts - true_values

    indexed_residuals = []
    for counter, sample_residuals in enumerate(residuals):
        indices = true_values_indices[counter]
        sample_residuals = pd.Series(index=indices, data=sample_residuals)
        indexed_residuals.append(sample_residuals)

    return indexed_residuals


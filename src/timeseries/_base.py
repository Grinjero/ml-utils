from timeseries.operations import align_series_list, align_series
import numpy as np

__all__ = [
    "extract_true_values_for_forecasts",
    "extract_true_values_for_forecast"
]


def extract_true_values_for_forecasts(true_values, sample_forecasts):
    if true_values.ndim == 1:
        true_values = align_series_list(sample_forecasts, true_values)

    true_values_indices = [true_sample_values.index for true_sample_values in true_values]
    true_values = np.asarray(true_values)
    return true_values_indices, true_values


def extract_true_values_for_forecast(true_values, sample_forecast):
    true_values = align_series(sample_forecast, true_values)
    return true_values


# def _preprocess_true_values(true_values, sample_forecasts):
#     if true_values.ndim == 1:
#         true_values = align_series_list(sample_forecasts, true_values)
#
#     true_values_indices = [true_sample_values.index[0] for true_sample_values in true_values]
#     true_values = np.asarray(true_values)
#     return true_values_indices, true_values
#
#
# def _preprocess_sample_forecasts(sample_forecasts):
#     if isinstance(sample_forecasts, list):
#         sample_forecasts = np.stack(sample_forecasts)
#
#     return sample_forecasts
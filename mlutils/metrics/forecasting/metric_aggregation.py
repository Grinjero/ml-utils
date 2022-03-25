import numpy as np
import pandas as pd

from mlutils.timeseries._operations import align_series_list


def _preprocess_true_values(true_values, sample_forecasts, skip_missing=False):
    if true_values.ndim == 1:
        true_values = align_series_list(sample_forecasts, true_values, skip_missing=skip_missing)

    true_values_indices = [true_sample_values.index[0] for true_sample_values in true_values]
    true_values = np.asarray(true_values)
    return true_values_indices, true_values


def _preprocess_sample_forecasts(sample_forecasts):
    if isinstance(sample_forecasts, list):
        sample_forecasts = np.stack(sample_forecasts)

    return sample_forecasts


def agg_forecast_origin_metrics(sample_forecasts, true_values, metric, skipna=False):
    """
    :param sample_forecasts: array like of shape (n_forecast_points, horizon_length)
    :param true_values: array like of shape (n_forecast_points + horizon_length) or same shape as :sample_forecasts
    :param metric: func with two same sized 1D arrays arguments (sample_forecasts, true_forecast_values) for evaluating sample_forecasts
    :param skipna: Skips evaluation of forecast origins with na true values or na forecast values as these can break most
        metric functions
    :return: metric for each forecast point (n_forecast_points) indexed by the true values associated with those forecast
        points
    """
    true_value_indices, true_values = _preprocess_true_values(true_values, sample_forecasts, skipna)
    stacked_forecasts = _preprocess_sample_forecasts(sample_forecasts)

    step_metrics = []
    step_index = []

    for forecast_point_index in range(0, stacked_forecasts.shape[0]):
        if skipna:
            if np.logical_or.reduce(np.isnan(true_values[forecast_point_index, :])) \
                    or np.logical_or.reduce(np.isnan(stacked_forecasts[forecast_point_index, :])):
                continue

        sample_true_values = true_values[forecast_point_index, :]
        forecast_metric = metric(sample_true_values, stacked_forecasts[forecast_point_index, :])

        step_metrics.append(forecast_metric)
        step_index.append(true_value_indices[forecast_point_index])

    return pd.Series(index=step_index, data=step_metrics)


def agg_forecast_origin_metrics_mask(sample_forecasts, true_values_series, metric, skipna=False, invalid_data_mask=None):
    """
    :param sample_forecasts: array like of shape (n_forecast_points, horizon_length)
    :param true_values: array like of shape (n_forecast_points + horizon_length) or same shape as :sample_forecasts
    :param metric: func with two same sized 1D arrays arguments (sample_forecasts, true_forecast_values) for evaluating sample_forecasts
    :param skipna: Skips evaluation of forecast origins with na true values or na forecast values as these can break most
        metric functions
    :param invalid_data_mask: Boolean pandas Series mask over the true_values series dictating which data points to skip in metric calculations
    :return: metric for each forecast point (n_forecast_points) indexed by the true values associated with those forecast
        points
    """
    true_value_indices, true_values_stacked = _preprocess_true_values(true_values_series, sample_forecasts, skipna)
    stacked_forecasts = _preprocess_sample_forecasts(sample_forecasts)

    if skipna:
        true_value_origin_na_mask = np.isnan(true_values_stacked).any(axis=1)
        forecasts_origin_na_mask = np.isnan(stacked_forecasts).any(axis=1)
        origin_na_mask = np.logical_or(true_value_origin_na_mask, forecasts_origin_na_mask)

    if invalid_data_mask is not None:
        stacked_invalid_mask = align_series_list(ts_list=sample_forecasts, alignment_series=invalid_data_mask)
        stacked_invalid_mask = np.isnan(np.stack(stacked_invalid_mask))
        origin_invalid_mask = stacked_invalid_mask.any(axis=1)

    step_metrics = []
    step_index = []

    for forecast_point_index in range(0, stacked_forecasts.shape[0]):
        if skipna:
            if origin_na_mask[forecast_point_index] is True:
                continue
        if invalid_data_mask is not None:
            if origin_invalid_mask[forecast_point_index] is True:
                continue

        sample_true_values = true_values_stacked[forecast_point_index, :]
        forecast_metric = metric(sample_true_values, stacked_forecasts[forecast_point_index, :])

        step_metrics.append(forecast_metric)
        step_index.append(true_value_indices[forecast_point_index])

    return pd.Series(index=step_index, data=step_metrics)


def agg_horizon_step_metrics(sample_forecasts, true_values, metric_func, skipna=False):
    """

    :param sample_forecasts: (n_forecast_points, forecast_horizon)
    :param true_values: (n_forecast_points + forecast_horizon)
    :param metric_func:
    :param skipna: skip na values found in sample_forecasts and observations during horizon step metric calculations
    :return: metric for each step in the horizon i.e. (forecast_horizon) sized array
    """

    _, true_values = _preprocess_true_values(true_values, sample_forecasts)
    stacked_forecasts = _preprocess_sample_forecasts(sample_forecasts)

    input_shape = stacked_forecasts.shape
    horizon_size = input_shape[1]
    horizon_metrics = []

    if skipna is True:
        nan_mask_true = np.isnan(true_values)
        nan_mask_forecast = np.isnan(stacked_forecasts)
        if (nan_mask_true.sum() > 0) or (nan_mask_forecast.sum() > 0):
            # Mask each place where there is either a Na true value or forecast
            for i in range(len(stacked_forecasts)):
                slice_obj = np.s_[i, None]

                sample_forecast_mask = np.logical_or(nan_mask_forecast[slice_obj], nan_mask_true[slice_obj])

                nan_mask_forecast[slice_obj] = sample_forecast_mask
                nan_mask_true[slice_obj] = sample_forecast_mask

            true_values = np.ma.masked_array(true_values, mask=nan_mask_true)
            stacked_forecasts = np.ma.masked_array(stacked_forecasts, mask=nan_mask_forecast)

    for horizon_step in range(horizon_size):
        predicted_horizon_steps = stacked_forecasts[:, horizon_step]
        true_horizon_steps = true_values[:, horizon_step]
        
        if skipna is True:
            predicted_horizon_steps = predicted_horizon_steps.compressed()
            true_horizon_steps = true_horizon_steps.compressed()
        metric = metric_func(predicted_horizon_steps, true_horizon_steps)
        horizon_metrics.append(metric)

    return horizon_metrics


def agg_horizon_step_metrics_masked(sample_forecasts, true_values_series, metric_func, skipna=False, invalid_data_mask=None):
    """

    :param sample_forecasts: (n_forecast_points, forecast_horizon)
    :param true_values: (n_forecast_points + forecast_horizon)
    :param metric_func:
    :param skipna: skip na values found in sample_forecasts and observations during horizon step metric calculations
    :param invalid_data_mask:  Boolean pandas Series mask over the true_values series dictating which data points to skip in metric calculations
    :return: metric for each step in the horizon i.e. (forecast_horizon) sized array
    """

    _, true_values_stacked = _preprocess_true_values(true_values_series, sample_forecasts)
    stacked_forecasts = _preprocess_sample_forecasts(sample_forecasts)

    input_shape = stacked_forecasts.shape
    horizon_size = input_shape[1]
    horizon_metrics = []

    if skipna is True:
        nan_mask_true = np.isnan(true_values_stacked)
        nan_mask_forecast = np.isnan(stacked_forecasts)

        if invalid_data_mask is not None:
            stacked_invalid_mask = align_series_list(ts_list=sample_forecasts, alignment_series=invalid_data_mask)
            stacked_invalid_mask = np.stack(stacked_invalid_mask)

        if (nan_mask_true.sum() > 0) or (nan_mask_forecast.sum() > 0) or invalid_data_mask is not None:
            # Mask each place where there is either a Na true value or forecast
            for i in range(len(stacked_forecasts)):
                slice_obj = np.s_[i, None]

                sample_forecast_mask = np.logical_or(nan_mask_forecast[slice_obj], nan_mask_true[slice_obj])
                if invalid_data_mask is not None:
                    sample_forecast_mask = np.logical_or(sample_forecast_mask, stacked_invalid_mask[slice_obj])

                nan_mask_forecast[slice_obj] = sample_forecast_mask
                nan_mask_true[slice_obj] = sample_forecast_mask

            true_values_stacked = np.ma.masked_array(true_values_stacked, mask=nan_mask_true)
            stacked_forecasts = np.ma.masked_array(stacked_forecasts, mask=nan_mask_forecast)

        print("Num masked in total: ", np.ma.count_masked(stacked_forecasts))

    for horizon_step in range(horizon_size):
        predicted_horizon_steps = stacked_forecasts[:, horizon_step]
        true_horizon_steps = true_values_stacked[:, horizon_step]

        if skipna is True:
            predicted_horizon_steps = predicted_horizon_steps.compressed()
            true_horizon_steps = true_horizon_steps.compressed()
        metric = metric_func(predicted_horizon_steps, true_horizon_steps)
        horizon_metrics.append(metric)

    return horizon_metrics


import numpy as np
import pandas as pd


#
#
# def _preprocess_true_values(true_values, sample_forecasts):
#
#
#
# def _preprocess_sample_forecasts(sample_forecasts):
    

def forecast_origin_metrics(sample_forecasts, true_values, metric, dropna=False):
    """
    :param sample_forecasts: array like of shape (n_forecast_points, horizon_length)
    :param true_values: array like of shape (n_forecast_points + horizon_length) or same shape as :sample_forecasts
    :param metric: func with two same sized 1D arrays arguments (sample_forecasts, true_forecast_values) for evaluating sample_forecasts
    :return: metric for each forecast point (n_forecast_points)
    """
    if isinstance(sample_forecasts, np.ndarray) is False:
        sample_forecasts = np.asarray(sample_forecasts)
    if true_values.ndim == 2 and isinstance(true_values, pd.DataFrame) and len(true_values.columns):
        true_values = true_values.iloc[:, 0]

    horizon_length = sample_forecasts.shape[1]
    step_metrics = []
    step_index = []
    if true_values.ndim == 2:
        for forecast_point_index in range(0, sample_forecasts.shape[0]):
            if dropna:
                if np.logical_or.reduce(true_values[forecast_point_index, :].isna()):
                    continue

            forecast_metric = metric(true_values[forecast_point_index, :], sample_forecasts[forecast_point_index, :])
            step_metrics.append(forecast_metric)

    elif true_values.ndim == 1:
        for forecast_point_index in range(0, sample_forecasts.shape[0]):
            # if np.logical_or.reduce(true_values[forecast_point_index:forecast_point_index + horizon_length].isna()):
            #     continue
            if dropna:
                if np.isnan(true_values[forecast_point_index:forecast_point_index + horizon_length]).sum() >= 1 or np.isnan(sample_forecasts[forecast_point_index, :]).sum() >= 1:
                     continue

            forecast_metric = metric(true_values[forecast_point_index:forecast_point_index + horizon_length], sample_forecasts[forecast_point_index, :])
            step_metrics.append(forecast_metric)
            step_index.append(true_values.index[forecast_point_index])

    else:
        raise ValueError("Illegal sample_forecasts or true value array shapes")

    if len(step_index) != 0:
        return pd.Series(index=step_index, data=step_metrics)
    return np.array(step_metrics)


def horizon_step_metrics(sample_forecasts, true_values, metric_func, dropna=False):
    """

    :param sample_forecasts: (n_forecast_points, forecast_horizon)
    :param true_values: (n_forecast_points + forecast_horizon)
    :param metric_func:
    :param dropna: Drop na sample_forecasts and observations
    :return: metric for each step in the horizon i.e. (forecast_horizon) sized array
    """
    if isinstance(true_values, list):
        true_values = np.array(true_values)
    if isinstance(sample_forecasts, list):
        sample_forecasts = np.stack(sample_forecasts)

    input_shape = sample_forecasts.shape
    horizon_size = input_shape[1]
    horizon_metrics = []

    if dropna is True:
        nan_mask_true = np.isnan(true_values)
        nan_mask_forecast = np.isnan(sample_forecasts)
        if (nan_mask_true.sum() > 0) or (nan_mask_forecast.sum() > 0):
            # Mask each place where there is either a Na true value or forecast
            for i in range(len(sample_forecasts)):
                true_slice = np.s_[i:i + horizon_size]
                forecast_slice = np.s_[i, None]

                sample_forecast_mask = np.logical_or(nan_mask_forecast[forecast_slice], nan_mask_true[true_slice])

                nan_mask_forecast[forecast_slice] = sample_forecast_mask
                nan_mask_true[true_slice] = sample_forecast_mask.flatten()

            true_values = np.ma.masked_array(true_values, mask=nan_mask_true)
            sample_forecasts = np.ma.masked_array(sample_forecasts, mask=nan_mask_forecast)

    for horizon_step in range(horizon_size):
        predicted_horizon_steps = sample_forecasts[:, horizon_step]
        true_horizon_steps = true_values[horizon_step:len(true_values) - (horizon_size - horizon_step + 1)]
        
        if dropna is True:
            predicted_horizon_steps = predicted_horizon_steps.compressed()
            true_horizon_steps = true_horizon_steps.compressed()
        metric = metric_func(predicted_horizon_steps, true_horizon_steps)
        horizon_metrics.append(metric)

    return horizon_metrics

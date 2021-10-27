import numpy as np
import pandas as pd


def step_forecast_metrics(forecasts, true_values, metric):
    """
    :param forecasts: array like of shape (n_forecast_points, horizon_length)
    :param true_values: array like of shape (n_forecast_points + horizon_length) or same shape as :forecasts
    :param metric: func with two same sized 1D arrays arguments (forecasts, true_forecast_values) for evaluating forecasts
    :return: metric for each forecast point (n_forecast_points)
    """
    if isinstance(forecasts, np.ndarray) is False:
        forecasts = np.asarray(forecasts)
    if true_values.ndim == 2 and isinstance(true_values, pd.DataFrame) and len(true_values.columns):
        true_values = true_values.iloc[:, 0]

    horizon_length = forecasts.shape[1]
    step_metrics = []
    step_index = []
    if true_values.ndim == 2:
        for forecast_point_index in range(0, forecasts.shape[0]):
            if np.logical_or.reduce(true_values[forecast_point_index, :].isna()):
                continue

            forecast_metric = metric(true_values[forecast_point_index, :], forecasts[forecast_point_index, :])
            step_metrics.append(forecast_metric)

    elif true_values.ndim == 1:
        for forecast_point_index in range(0, forecasts.shape[0]):
            # if np.logical_or.reduce(true_values[forecast_point_index:forecast_point_index + horizon_length].isna()):
            #     continue
            if np.isnan(true_values[forecast_point_index:forecast_point_index + horizon_length]).sum() >= 1 or np.isnan(forecasts[forecast_point_index, :]).sum() >= 1:
                 continue

            forecast_metric = metric(true_values[forecast_point_index:forecast_point_index + horizon_length], forecasts[forecast_point_index, :])
            step_metrics.append(forecast_metric)
            step_index.append(true_values.index[forecast_point_index])

    else:
        raise ValueError("Illegal forecasts or true value array shapes")

    if len(step_index) != 0:
        return pd.Series(index=step_index, data=step_metrics)
    return np.array(step_metrics)


def horizon_metric(sample_predictions, y_true, metric_func, dropna=False):
    """

    :param sample_predictions: (n_forecast_points, forecast_horizon)
    :param y_true: (n_forecast_points + forecast_horizon)
    :param metric_func:
    :param dropna: Drop na forecasts and observations
    :return: metric for each step in the horizon i.e. (forecast_horizon) sized array
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(sample_predictions, list):
        sample_predictions = np.stack(sample_predictions)

    input_shape = sample_predictions.shape
    horizon_size = input_shape[1]
    horizon_metrics = []

    if dropna:
        nan_mask_true = np.isnan(y_true)
        nan_mask_forecast = np.isnan(sample_predictions)
        if (nan_mask_true.sum() > 0) or (nan_mask_forecast.sum() > 0):
            # Mask each place where there is either a Na true value or forecast
            for i in range(len(sample_predictions)):
                true_slice = np.s_[i:i + horizon_size]
                forecast_slice = np.s_[i, None]

                sample_forecast_mask = np.logical_or(nan_mask_forecast[forecast_slice], nan_mask_true[true_slice])

                nan_mask_forecast[forecast_slice] = sample_forecast_mask
                print(sample_forecast_mask.shape)
                print(nan_mask_true[true_slice].shape)
                nan_mask_true[true_slice] = sample_forecast_mask.flatten()

            y_true = np.ma.masked_array(y_true, mask=nan_mask_true)
            sample_predictions = np.ma.masked_array(sample_predictions, mask=nan_mask_forecast)

    for horizon_step in range(horizon_size):
        predicted_horizon_steps = sample_predictions[:, horizon_step]
        true_horizon_steps = y_true[horizon_step:len(y_true) - (horizon_size - horizon_step + 1)]

        # nan_mask = np.isnan(true_horizon_steps)
        # nan_mask_forecast = np.isnan(sample_predictions)
        # if (nan_mask.sum() > 0) or (nan_mask_forecast.sum() > 0):
        #     true_horizon_steps = np.extract(-nan_mask, true_horizon_steps)
        #     predicted_horizon_steps = np.extract(-nan_mask, predicted_horizon_steps)

        metric = metric_func(predicted_horizon_steps, true_horizon_steps)
        horizon_metrics.append(metric)

    return horizon_metrics

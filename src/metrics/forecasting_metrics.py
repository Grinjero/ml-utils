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
    if true_values.ndim == 2:
        for forecast_point_index in range(0, forecasts.shape[0]):
            forecast_metric = metric(forecasts[forecast_point_index, :],
                                     true_values[forecast_point_index, :])
            step_metrics.append(forecast_metric)

    elif true_values.ndim == 1:
        for forecast_point_index in range(0, forecasts.shape[0]):
            forecast_metric = metric(forecasts[forecast_point_index, :],
                                     true_values[forecast_point_index:forecast_point_index + horizon_length])
            step_metrics.append(forecast_metric)

    else:
        raise ValueError("Illegal forecasts or true value array shapes")

    return np.array(step_metrics)


def horizon_metric(sample_predictions, y_true, metric_func):
    """

    :param sample_predictions: (n_forecast_points, forecast_horizon)
    :param y_true: (n_forecast_points + forecast_horizon)
    :param metric_func:
    :return: metric for each step in the horizon i.e. (forecast_horizon) sized array
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(sample_predictions, list):
        sample_predictions = np.stack(sample_predictions)

    input_shape = sample_predictions.shape
    horizon_size = input_shape[1]
    horizon_metrics = []
    for horizon_step in range(horizon_size):
        predicted_horizon_steps = sample_predictions[:, horizon_step]
        true_horizon_steps = y_true[horizon_step:len(y_true) - (horizon_size - horizon_step)]

        metric = metric_func(predicted_horizon_steps, true_horizon_steps)
        horizon_metrics.append(metric)

    return horizon_metrics

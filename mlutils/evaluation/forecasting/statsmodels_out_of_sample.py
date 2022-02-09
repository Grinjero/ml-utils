import numpy as np
import pandas as pd


def extract_exog_features(dataset, ex_features):
    if ex_features is None:
        return None
    ex = [dataset[feature_name].values for feature_name in ex_features]
    ex = np.stack(ex, axis=1)
    return ex


def step_by_step_forecast(model, endog, horizon_length, exog=None, only_predicted_mean=False):
    """
    Out of sample (as in the model was not fitted on this data) one step at a time forecasts

    :param model: fitted statsmodel model e.g. SARIMAX
    :param endog: endog values for the forecast period, expected to be after the period on which the model was fitted
    :type endog: pandas.Series
    :param exog: exog values for the forecast period, expected to be after the period on which the model was fitted
    :type exog: numpy.ndarray
    :param horizon_length: length of the forecasting horizon
    :type horizon_length: int
    :param only_predicted_mean: statsmodels get_prediction method returns various other information regarding the forecast,
        setting this to True returns only the forecast value
    :return: Step by step forecast over the given endog and exog sets,
        returns array in shape (len(endog) - horizon_length, horizon_length)
    """

    forecasting_start_index = 0
    try:
        extended_model = model.extend(endog, exog)
    except ValueError as e:
        raise ValueError("Model can't be extended " + str(e))

    forecasts = []
    for i in range(forecasting_start_index, len(endog) - horizon_length):
        sample_forecast = extended_model.get_prediction(start=endog.index[i], end=endog.index[i + horizon_length - 1],
                                                        dynamic=endog.index[i], exog=exog)

        if only_predicted_mean:
            sample_forecast = sample_forecast.predicted_mean

        forecasts.append(sample_forecast)

    return forecasts

def step_by_step_forecast_update(model, endog, horizon_length, update_step=7, exog=None, only_predicted_mean=False):
    """
    Out of sample (as in the model was not fitted on this data) one step at a time forecasts

    :param model: fitted statsmodel model e.g. SARIMAX
    :param endog: endog values for the forecast period, expected to be after the period on which the model was fitted
    :type endog: pandas.Series
    :param exog: exog values for the forecast period, expected to be after the period on which the model was fitted
    :type exog: numpy.ndarray
    :param horizon_length: length of the forecasting horizon
    :type horizon_length: int
    :param only_predicted_mean: statsmodels get_prediction method returns various other information regarding the forecast,
        setting this to True returns only the forecast value
    :return: Step by step forecast over the given endog and exog sets,
        returns array in shape (len(endog) - horizon_length, horizon_length)
    """

    forecasting_start_index = 0
    # try:
    #     extended_model = model.extend(endog, exog)
    # except ValueError as e:
    #     raise ValueError("Model can't be extended " + str(e))

    forecasts = []
    for i in range(forecasting_start_index, len(endog) - horizon_length):
        model.update(n_periods=horizon_length, X=exog[i:i+horizon_length])
        sample_forecast = extended_model.get_prediction(start=endog.index[i], end=endog.index[i + horizon_length - 1],
                                                        dynamic=endog.index[i], exog=exog)

        if only_predicted_mean:
            sample_forecast = sample_forecast.predicted_mean

        forecasts.append(sample_forecast)

    return forecasts

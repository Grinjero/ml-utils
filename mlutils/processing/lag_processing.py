import numpy as np
import pandas as pd

def lag_features(lags, series_to_lag):
    """
    :param lags: list of ints representing lags
    :param series_to_lag:
    :type series_to_lag: pd.Series
    :return:
    """

    feature_name = series_to_lag.name
    new_df = series_to_lag.to_frame()
    new_feature_names = []
    for lag in lags:
        lagged_feature_name = feature_name
        if lag > 0:
            lagged_feature_name += "+"
        lagged_feature_name += str(lag)

        new_feature_names.append(lagged_feature_name)
        lagged_series = series_to_lag.shift(-lag)
        new_df[lagged_feature_name] = lagged_series
    new_df = new_df.drop(columns=feature_name)
    return new_df, new_feature_names


def lag_features_on_group(group, lags, lag_column):
    new_df, new_feature_names = lag_features(lags, group[lag_column])
    group = group.join(new_df[new_feature_names])
    return group


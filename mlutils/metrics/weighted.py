import numpy as np
import pandas as pd


def series_weights_uniform(df: pd.DataFrame, weight_column, series_keys):
    """
    Calculates weights of series identified by `series_keys`. Weights are set to standard uniform weights (standard mean
        in other words)

    :param df: Contains all series
    :type df: pd.Dataframe
    :param weight_column: Name of the column from which weights are calculated
    :type weight_column: str
    :param series_keys: Columns which identify different series
    :type series_keys: list of str or a single str
    :return: series indexed by `series_keys` with values corresponding to weights of that series
    """
    groups = df.groupby(series_keys)
    ngroups = groups.ngroups
    return groups[weight_column].apply(lambda gr: 1 / ngroups)


def series_weights_by_mean(df: pd.DataFrame, weight_column, series_keys):
    """
    Calculates weights of series identified by `series_keys`. Weights are determined by the mean value of each group

    :param df: Contains all series
    :type df: pd.Dataframe
    :param weight_column: Name of the column from which weights are calculated
    :type weight_column: str
    :param series_keys: Columns which identify different series
    :type series_keys: list of str or a single str
    :return: series indexed by `series_keys` with values corresponding to weights of that series
    """
    groups = df.groupby(series_keys)
    group_means = groups[weight_column].mean()
    total_means = group_means.sum()
    return group_means / total_means


def series_weights_by_total(df: pd.DataFrame, weight_column, series_keys):
    """
    Calculates weights of series identified by `series_keys`. Weights are calculated by weighted averaging of total sums
        of each series

    :param df: Contains all series
    :type df: pd.Dataframe
    :param weight_column: Name of the column from which weights are calculated
    :type weight_column: str
    :param series_keys: Columns which identify different series
    :type series_keys: list of str or a single str
    :return: series indexed by `series_keys` with values corresponding to weights of that series
    """

    total_sum = df[weight_column].sum()
    return df.groupby(series_keys)[weight_column].apply(lambda gr: gr.sum() / total_sum)


def apply_weighted_metric(series_metrics_df, weights, metric_columns, new_columns_prefix="W", series_keys=None):
    """
    Applies the given weights to per-series metrics that are already calculated.

    :param series_metrics_df: df containing results for a single model
    :type series_metrics_df: pd.DataFrame
    :param weights: weights for each series metric
    :type weights: pd.DataFrame
    :param series_keys: Columns which identify different series
    :type series_keys: list of str or a single str
    :param metric_columns: columns names of each metric
    :return: weighted metric of metrics that have been calculated on individual series
    """

    if series_keys:
        indexed = series_metrics_df.set_index(series_keys)[metric_columns]
    else:
        # Assume the series is already indexed properly
        indexed = series_metrics_df[metric_columns]
    weighted_metrics = indexed.mul(weights, axis="index").sum()
    new_index_names = []
    for index in weighted_metrics.index:
        new_index_names.append(f"{new_columns_prefix}{index}")

    weighted_metrics.index = pd.Index(new_index_names, name="metric")
    return weighted_metrics

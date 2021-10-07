import numpy as np
import pandas as pd


def series_weights_by_mean(df: pd.DataFrame, weight_column, series_keys):
    """
    Calculates weights of series identified by `series_keys`. Weights are set to standard mean

    :param df: Contains all series
    :type df: pd.Dataframe
    :param weight_column: Name of the column from which weights are calculated
    :type weight_column: str
    :param series_keys: Columns which identify different series
    :type series_keys: list of str or a single str
    # :param series_index: Name of the columns by which each series will be indexed if necessary
    # :type series_index: str, optional
    :return: series indexed by `series_keys` with values corresponding to weights of that series
    """
    groups = df.groupby(series_keys)
    ngroups = groups.ngroups
    return groups.apply(lambda gr: 1 / ngroups)

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
    # :param series_index: Name of the columns by which each series will be indexed if necessary
    # :type series_index: str, optional
    :return: series indexed by `series_keys` with values corresponding to weights of that series
    """

    total_sum = df[weight_column].sum()
    return df.groupby(series_keys)[weight_column].apply(lambda gr: gr.sum() / total_sum)


def weighted_metric(series_metrics_df, weights, series_keys, metric_columns, new_columns_prefix="W"):
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

    indexed = series_metrics_df.set_index(series_keys)[metric_columns]
    weighted_metrics = indexed.mul(weights, axis="index").sum()
    new_index_names = []
    for index in weighted_metrics.index:
        new_index_names.append(f"{new_columns_prefix}{index}")

    weighted_metrics.index = pd.Index(new_index_names, name="metric")
    return weighted_metrics

#
# def series_range_percentage_weights(series):
#
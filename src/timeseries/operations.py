import pandas as pd
import numpy as np


def align_series_list(ts_list, alignment_series):
    """
    Extract subseries from alignment_series that are aligned with series from the ts_list.
    :param ts_list:
    :param alignment_series:
    :return: list of series extracted from alignment_series whose indices aligns with indices of series within ts_list
    """

    return [align_series(ts, alignment_series) for ts in ts_list]


# def align_series_indices_list(ts_list, alignment_series):
#

def align_series(series, alignment_series):
    """
    Extract a subserie from alignment_series whose index is aligned with series
    :param series:
    :param alignment_series:
    :return:
    """

    index = series.index
    return alignment_series.loc[index]


# def align_series_indices(series, alignment_series):
#     index = series.index
#     if alignment_series.index.isin(index)



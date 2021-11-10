import pandas as pd
import numpy as np


def align_series_list(ts_list, alignment_series, skip_missing=False):
    """
    Extract subseries from alignment_series that are aligned with series from the ts_list.
    :param ts_list:
    :param alignment_series:
    :param skip_missing: Skip elements from ts_list which are lacking corresponding indices in alignment_series
    :return: list of series extracted from alignment_series whose indices aligns with indices of series within ts_list
    """

    aligned_series = []
    for ts in ts_list:
        try:
            aligned_series.append(align_series(ts, alignment_series))
        except KeyError as e:
            if skip_missing is True:
                print(f"Missing corresponding values for \n{ts}")
                continue
            else:
                raise e
    return aligned_series


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



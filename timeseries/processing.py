import pandas as pd
import numpy as np


def element_checker(value, check_nan, check_inf, check_values=None):
    """
    Check if the given value or np.ndarray satisfies any of the given criteria
    :param value: value that is checked against following criteria
    :param check_inf:
    :param check_nan:
    :param check_values: checks if the number is in the given list of numbers or is equal to the given number
    :return: True if satisfies any
    """
    is_in = False
    if check_inf:
        is_in = is_in | np.isinf(value)
    if check_nan:
        is_in = is_in | np.isnan(value)
    if check_values:
        is_in = is_in | np.isin(value, check_values)

    return is_in


def remove_leading(series: pd.Series, is_nan_hole=True, is_inf_hole=False, hole_values=[]):
    first_valid_index = -1
    for i, element in enumerate(series):
        if not element_checker(element, is_nan_hole, is_inf_hole, hole_values):
            first_valid_index = i
            break

    if first_valid_index != -1:
        return series.drop(series.iloc[:first_valid_index].index)
    else:
        # nothing dropped
        return series


def remove_leading_na(series):
    return remove_leading(series, is_nan_hole=True)


def remove_following(series, is_nan_hole=True, is_inf_hole=False, hole_values=[]):
    first_invalid_index = len(series) + 1

    for index in range(len(series) - 1, 0, -1):
        element = series.iloc[index]

        if not element_checker(element, is_nan_hole, is_inf_hole, hole_values):
            first_invalid_index = index + 1
            break

    if first_invalid_index < len(series):
        return series.drop(series.iloc[first_invalid_index:].index)
    else:
        return series


def remove_following_na(series):
    return remove_following(series, is_nan_hole=True)

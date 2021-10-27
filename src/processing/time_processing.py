import math
from datetime import datetime, timedelta
import holidays
import numpy as np
import pandas as pd


def _extract_time_components(dataset, date_column="date"):
    dataset["date"] = pd.to_datetime(dataset[date_column])
    dataset["time"] = dataset[date_column].dt.time
    dataset["day"] = dataset[date_column].dt.day
    dataset["weekday"] = dataset[date_column].dt.weekday
    dataset["week"] = dataset[date_column].dt.week
    dataset["month"] = dataset[date_column].dt.month
    dataset["week_time"] = dataset[date_column].apply(
        lambda x: (x - (datetime(year=x.year, month=x.month, day=x.day) - timedelta(days=x.weekday()))))

    dataset["week_time_float"] = timedelta_to_float(dataset["week_time"])
    dataset["time_float"] = timedelta_to_float(dataset["time"])


# def timedelta_to_harmonic(timedelta_series, max_timedelta):
#
#
#

def timedelta_to_float(timedelta_series):
    return timedelta_series.seconds() / 3600


def _cyclic_hour_transformation(dataset):
    """
    sin and cos transformation is used to preserve the cyclical nature of hours
    """
    time_array = dataset["time"]
    normalized_time = np.array(
        [(60 * hours_minutes.hour + hours_minutes.minute) / (24 * 60) for hours_minutes in time_array])
    pied_time = normalized_time * 2 * math.pi
    time_sin, time_cos = np.sin(pied_time), np.cos(pied_time)
    polar_time = np.stack((time_sin, time_cos), axis=1)

    dataset["polar_time_sin"] = time_sin
    dataset["polar_time_cos"] = time_cos


def _cyclic_week_transformation(dataset):
    """
    sin and cos transformation is used to preserve the cyclical nature of weeks
    """
    week_time_array = dataset["week_time"]
    normalized_week_time = np.array(
        [week_time.total_seconds() / (7 * 24 * 3600) for week_time in week_time_array]
    )
    pied_week_time = normalized_week_time * 2 * math.pi
    week_sin, week_cos = np.sin(pied_week_time), np.cos(pied_week_time)

    dataset["polar_week_time_sin"] = week_sin
    dataset["polar_week_time_cos"] = week_cos


def _weekday_encoding(dataset):
    one_hot_encodings = pd.get_dummies(dataset.weekday, prefix='weekday')
    dataset[one_hot_encodings.columns.values.tolist()] = one_hot_encodings


def extract_time_features(dataset, **kwargs):
    _extract_time_components(dataset)
    _cyclic_hour_transformation(dataset)
    _cyclic_week_transformation(dataset)
    _weekday_encoding(dataset)


def is_working_day(date_column):
    croatia_holidays = holidays.HR()

    def check_element_working(date):
        weekday = date.weekday()
        return not ((weekday in [5, 6]) or (date in croatia_holidays))

    return date_column.map(check_element_working)


def dataframe_to_series(dataset, inplace):
    if inplace:
        dataset.set_index(keys="date", inplace=inplace, drop=False)
        dataset.sort_index(inplace=inplace)

    else:
        dataset = dataset.set_index(keys="date", inplace=inplace, drop=False)
        dataset = dataset.sort_index(inplace=inplace)

    dataset = dataset.resample("15min").ffill()
    return dataset

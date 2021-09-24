import pandas as pd
import numpy as np

from model import forecasting_interface


class BaselineBase(forecasting_interface.ForecastingModelInterface):
    def __init__(self, whole_series, target_column):
        self._series = whole_series
        self.target_column = target_column

    @property
    def series(self):
        return self._extract_series(self._series)

    def fit(self, train_data):
        self._series = self._extract_series(train_data)

    def _extract_series(self, outside_data):
        if self.target_column and isinstance(outside_data, pd.Series) is False:
            return outside_data[self.target_column]
        else:
            return outside_data

    def eval(self, test_data_df, horizon_length):
        test_series = self._extract_series(test_data_df)
        union = pd.concat([self.series, test_series], axis=0)
        reset_union = union.reset_index()
        self._series = reset_union.drop_duplicates(subset=union.index.name).set_index(union.index.name)

        forecasting_indices = test_series.index[:len(test_series) - horizon_length - 1]
        return [self.forecast(horizon_length, fp, index_or_label=True) for fp in forecasting_indices]


class NaiveSeasonal(BaselineBase):
    """
    Forecasted value is the value before seasonal_period of the currently forecasted index
    """
    def __init__(self, seasonal_period, whole_series: pd.Series=None, target_column=None):
        """
        :param whole_series: no worries, the model doesn't peek into the future
        :param seasonal_period:
        :param target_column: if given, whole_series is treated as a dataframe
        """
        super().__init__(whole_series, target_column)
        assert seasonal_period > 1, "Seasonal period must be greater than 1"

        self.seasonal_period = seasonal_period

    @property
    def name(self):
        return f"NaiveSeasonal_{self.seasonal_period}"

    def forecast(self, horizon_size, forecast_point_index, dynamic=True, index_or_label=False):
        """
        :param horizon_size: size of the forecasting horizon
        :param forecast_point_index: index after which the forecasting will begin
        :param dynamic: True for true forecasts,
        :param index_or_label: False to use the integer index of the series, True to use actual index (For example datetime if
        the series has a DateTimeIndex)
        :return: series containing the forecasts indexed by the same index of the original series
        """

        forecasted_values = []
        forecasted_indices = []
        if index_or_label is True:
            forecast_point_index = self.series.index.get_loc(forecast_point_index)

        assert forecast_point_index > self.seasonal_period, \
            "Can't start forecasting so early, forecast_point_index must be greater than seasonal_period"

        for i in range(1, min(horizon_size, self.seasonal_period) + 1):
            forecasted_value = self.series.iloc[forecast_point_index + i - self.seasonal_period]
            forecasted_index = self.series.index[forecast_point_index + i]

            forecasted_values.append(forecasted_value)
            forecasted_indices.append(forecasted_index)

        for i in range(self.seasonal_period + 1, horizon_size + 1):
            # only runs if n_steps is greater than the seasonal period
            forecasted_value = forecasted_values[-self.seasonal_period]
            forecasted_index = self.series.index[forecast_point_index + i]

            forecasted_values.append(forecasted_value)
            forecasted_indices.append(forecasted_index)

        return pd.Series(data=forecasted_values, index=forecasted_indices)


class Naive(BaselineBase):
    """
    Simply returns the last know value as the prediction for the whole horizon
    """

    def __init__(self, whole_series: pd.Series=None, target_column=None):
        """
        :param whole_series: no worries, the model doesn't peek into the future
        """
        super().__init__(whole_series, target_column)


    @property
    def name(self):
        return "Naive"


    def forecast(self, horizon_size, forecast_point_index, dynamic=True, index_or_label=False):
        """
        :param horizon_size: size of the forecasting horizon
        :param forecast_point_index: index after which the forecasting will begin
        :param dynamic: True for true forecasts,
        :param index_or_label: False to use the integer index of the series, True to use actual index (For example datetime if
        the series has a DateTimeIndex)
        :return: series containing the forecasts indexed by the same index of the original series
        """

        forecasted_values = []
        forecasted_indices = []
        if index_or_label is True:
            forecast_point_index = self.series.index.get_loc(forecast_point_index)

        assert forecast_point_index != 0, "No previous value to return"

        for i in range(1, horizon_size + 1):
            forecasted_value = self.series.iloc[forecast_point_index]
            forecasted_index = self.series.index[forecast_point_index + i]

            forecasted_values.append(forecasted_value)
            forecasted_indices.append(forecasted_index)

        return pd.Series(data=forecasted_values, index=forecasted_indices)


class MovingAverage(BaselineBase):
    """
    Forecasted value is the moving average of a sliding window. For longer horizons (larger than window_size) the sliding
     window calculates the moving average of previously calculated averages
    """
    def __init__(self, window_size, whole_series: pd.Series=None, target_column=None):
        """
        :param whole_series: no worries, the model doesn't peek into the future
        :param window_size: size of the moving average window
        """
        super().__init__(whole_series, target_column)
        assert window_size > 1, "Sliding window size must be greater than 1"

        self.window_size = window_size

    @property
    def name(self):
        return f"MovingAverage_{self.window_size}"

    def forecast(self, horizon_size, forecast_point_index, dynamic=True, index_or_label=False):
        """
        :param horizon_size: size of the forecasting horizon
        :param forecast_point_index: index after which the forecasting will begin
        :param dynamic: True for true forecasts,
        :param index_or_label: False to use the integer index of the series, True to use actual index (For example datetime if
        the series has a DateTimeIndex)
        :return: series containing the forecasts indexed by the same index of the original series
        """

        forecasted_values = []
        forecasted_indices = []
        if index_or_label is True:
            forecast_point_index = self.series.index.get_loc(forecast_point_index)

        assert forecast_point_index > self.window_size, "Forecast point index must be greater than the window size"

        mas = np.zeros(horizon_size + self.window_size)
        mas[:self.window_size] = self.series.iloc[forecast_point_index - self.window_size + 1:forecast_point_index + 1].values

        mas_index = self.window_size
        for i in range(1, horizon_size + 1):
            forecasted_index = self.series.index[forecast_point_index + i]
            mas[mas_index] = np.mean(mas[mas_index - self.window_size:mas_index])
            forecasted_value = mas[mas_index]

            forecasted_values.append(forecasted_value)
            forecasted_indices.append(forecasted_index)

            mas_index += 1

        return pd.Series(data=forecasted_values, index=forecasted_indices)


class SeasonalMovingAverage(BaselineBase):
    def __init__(self, window_size, seasonal_period, whole_series: pd.Series=None, target_column=None):
        super().__init__(whole_series, target_column)
        self.window_size = window_size
        self.seasonal_period = seasonal_period

    @property
    def name(self):
        return f"SeasonalMovingAverage_{self.window_size}_{self.seasonal_period}"

    def forecast(self, horizon_size, forecast_point_index, dynamic=True, index_or_label=False):
        """
        :param horizon_size: size of the forecasting horizon
        :param forecast_point_index: index after which the forecasting will begin
        :param dynamic: True for true forecasts,
        :param index_or_label: False to use the integer index of the series, True to use actual index (For example datetime if
        the series has a DateTimeIndex)
        :return: series containing the forecasts indexed by the same index of the original series
        """

        forecasted_values = []
        forecasted_indices = []
        if index_or_label is True:
            forecast_point_index = self.series.index.get_loc(forecast_point_index)

        assert forecast_point_index > self.window_size * self.seasonal_period, "Forecast point index must be greater than the window size * seasonal period"

        mas = np.zeros(horizon_size + self.window_size * self.seasonal_period)
        mas[:self.window_size * self.seasonal_period] = self.series.iloc[forecast_point_index - self.window_size * self.seasonal_period + 1:forecast_point_index + 1].values

        mas_index = self.window_size * self.seasonal_period
        for i in range(1, horizon_size + 1):
            forecasted_index = self.series.index[forecast_point_index + i]
            mas[mas_index] = np.mean(mas[mas_index - self.window_size * self.seasonal_period: mas_index: self.seasonal_period])

            forecasted_value = mas[mas_index]

            forecasted_values.append(forecasted_value)
            forecasted_indices.append(forecasted_index)

            mas_index += 1

        return pd.Series(data=forecasted_values, index=forecasted_indices)

if __name__ == "__main__":
    series = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    # seasonal_naive = NaiveSeasonal(series, seasonal_period=3)
    #
    # print(seasonal_naive.forecast(5, forecast_point_index=7))
    #
    # naive = Naive(series)
    # print(naive.forecast(5, forecast_point_index=4))

    # moving_average = MovingAverage(series, window_size=4)
    # print(moving_average.forecast(5, forecast_point_index=8))

    seasonal_ma = SeasonalMovingAverage(series, window_size=3, seasonal_period=3)
    print(seasonal_ma.forecast(4, forecast_point_index=13))

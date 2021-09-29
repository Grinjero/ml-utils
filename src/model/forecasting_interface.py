class ForecastingModelInterface:
    def fit(self, train_data_df):
        """
        Fit the model on the given data

        :param train_data_df:
        """
        pass

    def eval(self, test_data_df, horizon_length):
        """
        :param test_data_df:
        :param horizon_length:
        :return: list of forecasts for each forecast point in the given test_data_df. Each forecast is of length horizon_length
        """
        pass

    def forecast(self, forecast_point_index, horizon_length):
        """
        :param forecast_point_index: Starting point of the forecast
        :param horizon_length: Length of the forecasting horizon
        :return: list of horizon_length forecasted values starting from the given forecast_point_index
        """
        pass

    @property
    def name(self):
        """
        :return: Name of the given model identified by some of its hyperparameters or a predetermined name
        """
        return self.__class__

    def __str__(self):
        return self.name

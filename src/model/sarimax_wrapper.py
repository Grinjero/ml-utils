import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from evaluation.forecasting import extract_exog_features, step_by_step_forecast
from model import forecasting_interface


class SARIMAXWrapper(forecasting_interface.ForecastingModelInterface):
    def __init__(self, target_column, order, seasonal_order, exogenous_columns=None):
        self.target_column = target_column
        self.exogenous_columns = exogenous_columns

        self.order = order
        self.seasonal_order = seasonal_order

        self.model = None

    def fit(self, train_df):
        exogenous = extract_exog_features(train_df, self.exogenous_columns)
        model = SARIMAX(train_df[self.target_column], exog=exogenous, order=self.order, seasonal_order=self.seasonal_order)
        model_fit = model.fit()

        self.model = model_fit

    @property
    def name(self):
        return f"SARIMAX_({self.order[0]},{self.order[1]},{self.order[2]})_({self.seasonal_order[0]},{self.seasonal_order[1]},{self.seasonal_order[2]})x{self.seasonal_order[3]}"

    def eval(self, test_df, horizon_length):
        exogenous = extract_exog_features(test_df, self.exogenous_columns)

        if isinstance(test_df, pd.DataFrame):
            endog = test_df[self.target_column]
        else:
            endog = test_df

        forecasts = step_by_step_forecast(self.model, endog, horizon_length, exog=exogenous,
                                          only_predicted_mean=True)

        return forecasts
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from evaluation.forecasting import extract_exog_features, step_by_step_forecast
from model import forecasting_interface


class SARIMAXWrapper(forecasting_interface.ForecastingModelInterface):
    def __init__(self, target_column, order, seasonal_order, exogenous_columns=None):
        """
        Wrapper for the statsmodels.tsa.statespace.sarimax.SARIMAX model so it can be more easily used alongside other
        other methods

        :param target_column: name of the column that is the targeted forecasted value
        :param order: (p, d, q) of the SARIMAX model
        :param seasonal_order: seasonal (P, D, Q, S) of the SARIMAX model; S is the seasonal length
        :param exogenous_columns: columns that will be used as exogenous inputs for the model. If provided make sure that
            the fit and eval function input DataFrames contain given columns.
        """
        super().__init__()

        self.target_column = target_column
        self.exogenous_columns = exogenous_columns

        self.order = order
        self.seasonal_order = seasonal_order

        self.model = None

    def fit(self, train_df):
        exogenous = extract_exog_features(train_df, self.exogenous_columns)
        model = SARIMAX(train_df[self.target_column], exog=exogenous, order=self.order,
                        seasonal_order=self.seasonal_order)
        model_fit = model.fit()

        self.model = model_fit

    @property
    def name(self):
        return "SARIMAX"

    def hyperparameters(self):
        return {
            "target": self.target_column,
            "exogenous": self.exogenous_columns,
            "order": self.order,
            "seasonal_order": self.seasonal_order
        }

    def eval(self, test_df, horizon_length):
        exogenous = extract_exog_features(test_df, self.exogenous_columns)

        if isinstance(test_df, pd.DataFrame):
            endog = test_df[self.target_column]
        else:
            endog = test_df

        forecasts = step_by_step_forecast(self.model, endog, horizon_length, exog=exogenous,
                                          only_predicted_mean=True)

        return forecasts

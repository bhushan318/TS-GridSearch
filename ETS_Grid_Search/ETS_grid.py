"""ETS grid is a class to find best trend and seasonality type for given time series"""
import warnings
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

warnings.filterwarnings("ignore")


class ETSGRID:

    """ETS grid is a class to find best trend and seasonality type for given time series"""

    def __init__(self, train, test, seasonal_period):
        """
        :param train: time series
        :param test: time series
        :param seasonal_period: seasonal period for given time series Ex: 4 for quarterly data
        or 7 for daily data with a weekly cycle. or 12 for monthly data
        """

        self.train = train
        self.test = test
        self.seasonal_period = seasonal_period

    def ets_singlerun(self, trend_type, seasonal_type):
        """for given trend and seasonal type model will be fitted and its results
        are returned if particular combination of parameters are not suitable then
        function will return nan values"""

        try:
            model = ExponentialSmoothing(
                endog=self.train,
                trend=trend_type,
                damped=True,
                seasonal=seasonal_type,
                seasonal_periods=self.seasonal_period,
                dates=None,
                freq=None,
                missing="none",
            ).fit()
            aic = model.aic
            bic = model.bic
            error_mse_fitting = round(
                mean_squared_error(self.train, model.fittedvalues)
            )
            error_mae_fitting = round(
                mean_absolute_error(self.train, model.fittedvalues)
            )
            error_mape_fitting = (
                round(mean_absolute_percentage_error(self.train, model.fittedvalues))
                * 100
            )

            if self.test.empty:
                error_mse_prediction = np.nan
                error_mae_prediction = np.nan
                error_mape_prediction = np.nan

            else:
                predictions = model.predict(
                    start=len(self.train), end=len(self.train) + len(self.test) - 1
                )
                error_mse_prediction = round(mean_squared_error(self.test, predictions))
                error_mae_prediction = round(
                    mean_absolute_error(self.test, predictions)
                )
                error_mape_prediction = (
                    round(mean_absolute_percentage_error(self.test, predictions)) * 100
                )

            return (
                trend_type,
                seasonal_type,
                self.seasonal_period,
                aic,
                bic,
                error_mse_fitting,
                error_mae_fitting,
                error_mape_fitting,
                error_mse_prediction,
                error_mae_prediction,
                error_mape_prediction,
            )
        except:
            return (
                trend_type,
                seasonal_type,
                self.seasonal_period,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

    def etsgridsearch(self):
        """for given data models are fitted based on trend and seasonal type"""

        trend_seasonal = pd.DataFrame(
            {
                "trend": ["add", "add", "mul", "mul"],
                "seasonal": ["add", "mul", "mul", "add"],
            }
        )

        result_table_ets = pd.DataFrame(
            np.vectorize(self.ets_singlerun)(
                trend_seasonal["trend"], trend_seasonal["seasonal"]
            )
        ).transpose()
        result_table_ets.columns = [
            "Trend",
            "Seasonal",
            "seasonal_period",
            "AIC",
            "BIC",
            "error_mse_fitting",
            "error_mae_fitting",
            "error_mape_fitting",
            "error_mse_prediction",
            "error_mae_prediction",
            "error_mape_prediction",
        ]

        result_table_ets = result_table_ets.sort_values(
            by="AIC", ascending=True
        ).reset_index(drop=True)
        result_table_ets_1 = result_table_ets[
            ~result_table_ets.Trend.str.contains("Fals")
        ]
        return result_table_ets_1

""" SARIMA grid is a class to find best p, d, q  and P, D, Q for given time series"""
from itertools import product
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import statsmodels.api as sm
import pandas as pd

warnings.filterwarnings("ignore")


class SARIMAGRID:

    """SARIMA grid is a class to find best p, d, q  and P, D, Q for given time series"""

    def __init__(self, train, test, seasonal_period, max_value=4):
        """User need to provide train and test data which should be a univariate sample and
        seasonal period Ex: 4 for quarterly data or 7 for daily data with a weekly cycle.
        or 12 for monthly data. for SARIMA hyper parameters p,d,q, P, D and Q the Max value to be
        provided as max_value by default 4 is considered"""

        self.train = train
        self.test = test
        self.seasonal_period = seasonal_period
        self.max_value = max_value

    def sarima_singlerun(self, p, d, q, ps, ds, qs, s):
        """for given p,d,q, ps, ds and qs values model will be fitted and its results
        are returned if particular combination of parameters are not sutiable then
        function will return nan values"""

        try:
            model = sm.tsa.statespace.SARIMAX(
                endog=self.train,
                order=(p, d, q),
                seasonal_order=(ps, ds, qs, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=-1)
            aic = model.aic
            bic = model.bic
            error_mse_fitting = round(
                mean_squared_error(self.train, model.fittedvalues)
            )
            error_mae_fitting = round(
                mean_absolute_error(self.train, model.fittedvalues)
            )
            error_mape_fitting = round(
                mean_absolute_percentage_error(self.train, model.fittedvalues) * 100
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
                error_mape_prediction = round(
                    mean_absolute_percentage_error(self.test, predictions) * 100
                )

            return (
                p,
                d,
                q,
                ps,
                ds,
                qs,
                s,
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
                p,
                d,
                q,
                ps,
                ds,
                qs,
                s,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

    def sarimagridsearch(self):
        """for given data models are fitted based on trend and seasonal type"""

        p = d = q = ps = ds = qs = range(0, self.max_value)
        s = range(self.seasonal_period, self.seasonal_period + 1)
        parameters = product(p, d, q, ps, ds, qs, s)
        parameters_list = list(parameters)
        data_frame = pd.DataFrame(parameters_list)
        data_frame.columns = ["p", "d", "q", "P", "D", "Q", "s"]
        result_table_sarima = pd.DataFrame(
            np.vectorize(self.sarima_singlerun)(
                data_frame["p"],
                data_frame["d"],
                data_frame["q"],
                data_frame["P"],
                data_frame["D"],
                data_frame["Q"],
                data_frame["s"],
            )
        ).transpose()
        result_table_sarima.columns = [
            "p",
            "d",
            "q",
            "P",
            "D",
            "Q",
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
        result_table_sarima = result_table_sarima.sort_values(
            by="AIC", ascending=True
        ).reset_index(drop=True)
        return result_table_sarima

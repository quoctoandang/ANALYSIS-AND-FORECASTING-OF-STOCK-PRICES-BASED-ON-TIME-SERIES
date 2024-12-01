import warnings
import itertools
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
from datetime import datetime
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from seaborn import histplot
import time
import joblib



class Sarima_predictions():
    def __init__(self):
        
        pass

    def read_data(self, path):
        data = pd.read_csv(path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data.fillna(0, inplace=True)
        return data

    def split_data(self, data, split_ratio=0.8):
        cut_off = int(len(data) * split_ratio)
        train = data[:cut_off]
        test = data[cut_off:]
        X_train = train.drop(['close'], axis=1)
        X_test = test.drop(['close'], axis=1)
        y_train = train['close']
        y_test = test['close']
        return X_train, X_test, y_train, y_test


    def find_best_model_sarimax(self, x_train, y_train):
        model  = auto_arima(y=y_train,
                    X=x_train,
                    m = 7,
                    max_P=5,
                    max_Q=5,
                    seasonal=True,
                    test='adf',
                    alpha=0.05,              
                    d=None,                  
                    D=None,
                    error_action='ignore',
                    suppress_warnings=True, 
                    trace=True,
                    scoring='mse')
        return model.order, model.seasonal_order
    
    def fit(self, x_train, y_train):
        orders, seasonal_orders = self.find_best_model_sarimax(x_train, y_train)

        model = SARIMAX(
            endog=y_train,
            exog=x_train,
            order=orders,
            seasonal_order=seasonal_orders,
            freq='D',
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False,
            mle_regression=True
        )
        model_fit = model.fit(disp=0)
        print("Model's AIC = {:.4f}".format(model_fit.aic))
        print(model_fit.summary())

        model_filename = "../model_save/sarimax_model.pkl"
        joblib.dump(model_fit, model_filename)

        return model_fit, orders, seasonal_orders
    
    def diagnostics(self,y_test, forecasts):
        resid = y_test - forecasts

        # plot residuals
        plt.figure(figsize=(6, 3))

        histplot(x=resid, kde=True, bins=50)
        plt.axvline(x=resid.mean(), color='red') # plot mean value
        plt.title('Close Price Residuals: SARIMAX')
        plt.xlabel(None)
        plt.ylabel(None)

        plt.show()
            
    def predict(self, model, y_train, x_train, x_test, y_test):
        forecasts = model.predict(start=y_test.index[0], end=y_test.index[-1], exog=x_test, dynamic=True)
        result = pd.concat([y_test, forecasts], axis=1).rename(columns={'close': 'true', 'predicted_mean': 'pred'})
        ticker = 'forecasts'
        path = "../results/sarimax/{}.csv".format(ticker)
        result.to_csv(path, index=True)

        return forecasts

    def cross_val_sarimax(self, data, orders, seasonal_orders, k=6, test_data=True):
        '''
        Returns k-fold cross-validation score (RMSE).

        '''
        tscv = TimeSeriesSplit(n_splits=k)
        cv_scores = []
        
        for train_index, test_index in tscv.split(data):
            # set endogenous & exogenous variables
            cv_endog_train, cv_endog_test = data.iloc[train_index, -1:], data.iloc[test_index, -1:]
            cv_exog_train, cv_exog_test = data.iloc[train_index, :-1], data.iloc[test_index, :-1]
            
            mod = SARIMAX(
                endog=cv_endog_train,
                exog=cv_exog_train,
                order=orders,
                seasonal_order=seasonal_orders,
                freq='D',
                trend=None,
                enforce_stationarity=False,
                enforce_invertibility=False,
                mle_regression=True
            ).fit()
            
            # get scores & append to list
            forecasts = mod.predict(start=cv_endog_test.index[0], end=cv_endog_test.index[-1], exog=cv_exog_test, dynamic=True)
            rmse = np.sqrt(mean_squared_error(cv_endog_test, forecasts))
            cv_scores.append(rmse)
            
        return np.mean(cv_scores)
    
    def evulation(self, data, orders, seasonal_orders,  y_test, forecasts):

        rmse = np.sqrt(mean_squared_error(y_test, forecasts))
        cv_rmse = self.cross_val_sarimax(data, orders, seasonal_orders)
        r2 = r2_score(y_test, forecasts)
        mae = mean_absolute_error(y_test, forecasts)
        mape = mean_absolute_percentage_error(y_test, forecasts)

        print('Testing performance:')
        print('--------------------')
        print('RMSE: {:.4f}'.format(rmse))
        print('6-fold CV: {:.4f}'.format(cv_rmse))
        print('R2: {:.4f}'.format(r2))
        print('MAE: {:.4f}'.format(mae))
        print('MAPE: {:.4f}%'.format(mape))
        results = {
        'Metric': ['RMSE', '6-fold CV', 'R2', 'MAE', 'MAPE'],
        'Value': [rmse, cv_rmse, r2, mae, mape]
        }
        df = pd.DataFrame(results)
        path = "../results/sarimax/evulation.csv"
        df.to_csv(path, index=False)
        print(f'Results saved to {path}')

    def plot_result(self, y_train,y_test, forecasts):
        fig, axes = plt.subplots(2, 1, sharex=False, figsize=(9, 6), tight_layout=True, gridspec_kw={'height_ratios': [1.5, 1]})
        axes[0].plot(y_train, label='train')
        axes[0].plot(y_test, linewidth=1.5, label='test')
        axes[0].plot(forecasts, color='green', label='forecast')
        axes[0].set_title('Close Price Estimations: SARIMAX')
        axes[0].legend()

        axes[1].plot(y_test, linewidth=1.5, label='test')
        axes[1].plot(forecasts, label='forecast')
        axes[1].set_xlim(y_test.index[0], y_test.index[-1])
        axes[1].set_ylim(min(y_test.min(), forecasts.min()) - 10, max(y_test.max(), forecasts.max()) + 10)
        axes[1].legend()

        plt.show()

       

    def baseline_sarimax(self, path):
        data = self.read_data(path)
        X_train, X_test, y_train, y_test = self.split_data(data)
        print('Training data:', X_train.shape, y_train.shape)
        model_fit, order, seasonal_order = self.fit(X_train, y_train)
        forecasts = self.predict(model_fit, y_train, X_train, X_test, y_test)
        self.plot_result(y_train,y_test, forecasts)
        self.diagnostics(y_test, forecasts)
        print('Model evaluation Sarimax 1:')
        self.evulation(data, order, seasonal_order, y_test, forecasts)

if __name__ == '__main__':
    print("-----------------RUN-----------------")
    path = '../data/data_train_model.csv'
    _s = Sarima_predictions()
    _s.baseline_sarimax(path)

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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold


class Residual_predictions():
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
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
        return X_train, X_test, y_train, y_test, train, test


    def find_best_model_sarimax(self, x_train, y_train):
        model  = auto_arima(y=y_train,
                    X=x_train,
                    m = 7,
                    start_P=0,
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
        residual = model_fit.resid

        return residual
    
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
        X_train, X_test, y_train, y_test, train, test = self.split_data(data)
        print('Training data:', X_train.shape, y_train.shape)
        residual = self.fit(X_train, y_train)
        scaled_train = self.scaler.fit_transform(train)
        scaled_test = self.scaler.fit_transform(test)
        look_back = 50
        X_train = []
        y_train = []
        for i in range(look_back, len(train)):
            X_train.append(residual[i-look_back:i, 0])
            y_train.append(train[i])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1],1 )))
        model.add(LSTM(50, return_sequence = False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer=self.lstm_params['optimizer'], loss='mean_squared_error')
        lstm_history = model.fit(
                        x=X_train,
                        y=y_train,
                        epochs=50,
                        batch_size=10,
                        verbose=1,
                        callbacks=None
                        )
        test_data_scaled = self.scaler.transform(test.reshape(-1,1))
        X_test = []
        y_test = []
        for i in range(look_back, len(test_data_scaled)):
            X_test.append(test_data_scaled[i-look_back:i, 0])
            y_test.append(test[i])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predict = model.predict(X_test)
        predict = self.scaler.inverse_transform(predict)

     

if __name__ == '__main__':
    print("-----------------RUN-----------------")
    path = '../data/data_train_model.csv'
    _s = Residual_predictions()
    _s.baseline_sarimax(path)

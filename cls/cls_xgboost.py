import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from seaborn import histplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
import joblib


class XGboostConfig():
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.xgb_param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate' : [0.001, 0.01],
        #'reg_alpha': [1, 10, 100],          
        'n_estimators': [1000]
        }

    def read_data(self, path):
        """Đọc dữ liệu từ file CSV."""
        data = pd.read_csv(path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data.fillna(0, inplace=True)
        return data
    
    def split_data(self, data, split_ratio=0.8):
        cut_off = int(len(data) * split_ratio)
        train = data[:cut_off]
        test = data[cut_off:]
        train_scale = self.scaler.fit_transform(train)
        test_scale = self.scaler.fit_transform(test)
        X = data.iloc[:, :-1]
        y = data['close']
        X_train, y_train = train_scale[:, :-1], train_scale[:, -1].reshape(-1, 1)
        X_test, y_test = test_scale[:, :-1], test_scale[:, -1].reshape(-1, 1)

        return train, test, X_train, X_test, y_train, y_test
    
    def cross_validation(self,X_train, y_train, estimator, k=6):
        '''
        Returns k-fold cross-validation score (RMSE).
        '''
        k_folds = KFold(n_splits=k, shuffle=False)
        cv_scores = cross_val_score(estimator, X_train, y_train.squeeze(), cv=k_folds, scoring='neg_root_mean_squared_error')
        
        return abs(cv_scores).mean()
    
    def plot_result(self, train, test, y_test, pred, true):
        fig, axes = plt.subplots(2, 1, sharex=False, figsize=(9, 6), tight_layout=True, gridspec_kw={'height_ratios': [1.5, 1]})

        axes[0].plot(train['close'], label='train')
        axes[0].plot(test['close'], linewidth=1.5, label='test')
        axes[0].plot(test.index, pred, color='green', label='forecast')
        axes[0].set_title('Close Price Estimations: XGBoosting Regression')
        axes[0].legend()

        # zoomed view
        axes[1].plot(test.index, true, linewidth=1.5, label='test')
        axes[1].plot(test.index, pred, label='forecast')
        axes[1].set_xlim(test.index[0], test.index[-1]) 
        axes[1].set_ylim(min(y_test.min(), pred.min()), max(y_test.max(), pred.max()))
        axes[1].legend()

        plt.show()

    def plot_residuals(self, residuals):
        plt.figure(figsize=(6, 3))

        histplot(x=residuals, kde=True, bins=50)
        plt.axvline(x=residuals.mean(), color='red') # plot mean value
        plt.title('Close Price Residuals: XGBoosting Regression')
        plt.xlabel(None)
        plt.ylabel(None)

        plt.show()
    
    def baseline_XGboost(self, path):
        # Lấy dữ liệu
        data = self.read_data(path)
        # split data 
        train, test, X_train, X_test, y_train, y_test  = self.split_data(data)
        xgb_model = GridSearchCV(estimator=XGBRegressor(), param_grid=self.xgb_param_grid, verbose=1)
        xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        xgb_kwargs = xgb_model.best_params_
        print('\nBest parameters for XGBoosting:\n', xgb_kwargs)
        model_path = "../model_save/xgboost_model.joblib"
        joblib.dump(xgb_model.best_estimator_, model_path)
        xgb_predictions = xgb_model.predict(X_test).reshape(-1, 1)
        xgb_pred_copies = np.repeat(xgb_predictions, test.shape[1], axis=-1)

        xgb_pred = self.scaler.inverse_transform(xgb_pred_copies)[:, 0]
        true_copies = np.repeat(y_test, test.shape[1], axis=-1)
        true = self.scaler.inverse_transform(true_copies)[:, 0]
        # display trues vs. forecasts
        xgb_estimator = XGBRegressor(**xgb_kwargs)
        
        rmse = np.sqrt(mean_squared_error(true, xgb_pred))
        cv_rmse =  self.cross_validation(X_train, y_train, xgb_estimator)
        r2 = r2_score(true, xgb_pred)
        mae = mean_absolute_error(true, xgb_pred)
        mape = mean_absolute_percentage_error(true, xgb_pred)

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
        path = "../results/xgboots/evulations.csv"
        df.to_csv(path, index=False)
        print(f'Results saved to {path}')

        xgb_residuals = true - xgb_pred
        self.plot_result(train, test, y_test, xgb_pred, true)
        self.plot_residuals(xgb_residuals)

if __name__ == '__main__':
    print("-----------------RUN-----------------")
    path = '../data/data_train_model.csv'
    _s = XGboostConfig()
    _s.baseline_XGboost(path)


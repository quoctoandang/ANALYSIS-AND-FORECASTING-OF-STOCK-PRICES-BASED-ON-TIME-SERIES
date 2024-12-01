import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from seaborn import histplot



class LSTMConfig():
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.prediction_step = 50
        self.lstm_params = {
            'loss': 'mean_squared_error',
            'metrics': ['mean_absolute_error'],
            'optimizer': 'adam',
            'lstm_units': 64,
            'epochs': 50,
            'batch_size': 10,
            'validation_split': 0.1
        }

    def read_data(self, path):
        """Đọc dữ liệu từ file CSV."""
        data = pd.read_csv(path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data.fillna(0, inplace=True)
        return data
    
    def split_and_transform_data(self, data, split_ratio=0.8):
        cut_off = int(len(data) * split_ratio)
        train = data[:cut_off]
        test = data[cut_off:]
        scaled_train = self.scaler.fit_transform(train)
        print('Train shape:', scaled_train.shape)
        scaled_test = self.scaler.fit_transform(test)
        print('Test shape:', scaled_test.shape)
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        test_length = len(data) - cut_off
        prediction_step = 50
        for i in range(prediction_step, cut_off):
            X_train.append(scaled_train[i - prediction_step:i, 0:scaled_train.shape[1]])
            y_train.append(scaled_train[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        print('X_train shape :', X_train.shape)
        print('y_train shape :', y_train.shape)

        for i in range(prediction_step, test_length):
            X_test.append(scaled_test[i - prediction_step:i, 0:scaled_test.shape[1]])
            y_test.append(scaled_test[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        y_test = np.reshape(y_test, (-1, 1))
        
        print('X_test shape :', X_test.shape)
        print('y_test shape :', y_test.shape)

        return train, test, X_train, X_test, y_train, y_test
    
    def get_lstm_model(self, X_train):
        '''
        Returns LSTM model.
        '''
        # set architecture
        model = Sequential(
            [
                # 1st LSTM layer
                LSTM(units=self.lstm_params['lstm_units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                
                # 2nd layer
                LSTM(units=self.lstm_params['lstm_units'], return_sequences=False),
                
                Dense(units=32),
                
                # output layer
                Dense(units=1)
                #Dense(units=y_train.shape[1]) # 1-size output
                
            ], name='LSTM_model'
        )
        
        model.compile(optimizer=self.lstm_params['optimizer'], loss=self.lstm_params['loss'])
        
        return model
    
    def fit(self, lstm_model, X_train, y_train):
        lstm_history = lstm_model.fit(
                        x=X_train,
                        y=y_train,
                        epochs=self.lstm_params['epochs'],
                        batch_size=self.lstm_params['batch_size'],
                        verbose=1,
                        callbacks=None
                        )

        model_path = '../model_save/lstm_model.h5'  
        lstm_model.save(model_path)
        
        return lstm_history
    
    def predict(self, lstm_model, X_test, test):
        lstm_predictions = lstm_model.predict(X_test)
        prediction_copies = np.repeat(lstm_predictions, test.shape[1], axis=-1)
        lstm_pred = self.scaler.inverse_transform(prediction_copies)[:, 0]
        return lstm_pred
    
    def plot_result(self ,train, test,y_test, lstm_pred, true):
        fig, axes = plt.subplots(2, 1, sharex=False, figsize=(9, 6), tight_layout=True, gridspec_kw={'height_ratios': [1.5, 1]})

        axes[0].plot(train['close'], label='train')
        axes[0].plot(test['close'], linewidth=1.5, label='test')
        axes[0].plot(test[self.prediction_step:].index, lstm_pred, color='green', label='forecast')
        axes[0].set_title('Close Price Estimations: LSTM')
        axes[0].legend()

        # zoomed view
        axes[1].plot(test[self.prediction_step:].index, true, linewidth=1.5, label='test')
        axes[1].plot(test[self.prediction_step:].index, lstm_pred, label='forecast')
        axes[1].set_xlim(y_test.index[0], test.index[-1]) 
        axes[1].set_ylim(min(y_test.min(), lstm_pred.min()), max(y_test.max(), lstm_pred.max()))
        axes[1].legend()

        plt.show()
    
    def cross_val_lstm(self, inputs, targets, X_train, k=6):
        '''
        Returns k-fold cross-validation score (RMSE) for LSTM.
        '''
        lstm_cv_scores = []
        kfold = KFold(n_splits=k, shuffle=False)
            
        fold_num = 1
        for train, test in kfold.split(inputs, targets):
            lstm_mod = self.get_lstm_model(X_train)
            
            print('-------------------------------------------------------------')
            print(f'Training for {fold_num}-fold:')
            
            lstm_mod.fit(
                x=inputs[train],
                y=targets[train],
                epochs=self.lstm_params['epochs'],
                batch_size=self.lstm_params['batch_size'],
                verbose=0)
            
            predictions = lstm_mod.predict(inputs[test], verbose=0)
            
            rmse = np.sqrt(((targets[test] - predictions)**2).mean())
            print('RMSE: {:.4f}'.format(rmse))
            lstm_cv_scores.append(rmse)
                
            # increase fold number
            fold_num = fold_num + 1
        return lstm_cv_scores
    
    def plot_LSTM_loss(self, lstm_history):
        plt.figure(figsize=(6, 3))
        plt.plot(range(1, self.lstm_params['epochs'] + 1), lstm_history.history['loss'])
        plt.title('LSTM Loss')
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        
        plt.show()
    
    def plot_Residuals(self, lstm_resid):
        plt.figure(figsize=(6, 3))

        histplot(x=lstm_resid, kde=True, bins=50)
        plt.axvline(x=lstm_resid.mean(), color='red') # plot mean value
        plt.title('Close Price Residuals: LSTM')
        plt.xlabel(None)
        plt.ylabel(None)

        plt.show()
    
    def baseline_LSTM(self, path):
        # Lấy dữ liệu
        data = self.read_data(path)
        # split data 
        train, test, X_train, X_test, y_train, y_test = self.split_and_transform_data(data)
        print('X_train:', np.isnan(X_train).any())
        print('y_train:', np.isnan(y_train).any())
        print('X_test:', np.isnan(X_test).any())
        print('y_test:', np.isnan(y_test).any())
        lstm_model = self.get_lstm_model(X_train)
        lstm_history = self.fit(lstm_model, X_train, y_train )
        lstm_pred = self.predict(lstm_model, X_test, test)

        true_copies = np.repeat(y_test, test.shape[1], axis=-1)
        true = self.scaler.inverse_transform(true_copies)[:, 0]

        # join arrays
        inputs = np.concatenate((X_train, X_test), axis=0)
        targets = np.concatenate((y_train, y_test), axis=0)
        lstm_cv_scores = self.cross_val_lstm(inputs, targets, X_train)
        
        rmse = np.sqrt(mean_squared_error(true, lstm_pred))
        cv_rmse = sum(lstm_cv_scores)/6
        r2 = r2_score(true, lstm_pred)
        mae = mean_absolute_error(true, lstm_pred)
        mape = mean_absolute_percentage_error(true, lstm_pred)
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
        path = "../results/lstm/evulation.csv"
        df.to_csv(path, index=False)
        print(f'Results saved to {path}')

        lstm_resid = true - lstm_pred
        # visualize
        self.plot_LSTM_loss(lstm_history)
        self.plot_result(train,test, y_test, lstm_pred, true)
        self.plot_Residuals(lstm_resid)

if __name__ == '__main__':
    print("-----------------RUN-----------------")
    path = '../data/data_train_model.csv'
    _s = LSTMConfig()
    _s.baseline_LSTM(path)
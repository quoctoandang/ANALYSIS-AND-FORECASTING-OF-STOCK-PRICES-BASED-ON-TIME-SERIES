import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
import joblib

class RfConfig:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.prediction_step = 50
        self.rf_params = {
            'n_estimators': 300,  # Tăng số lượng cây do có nhiều dữ liệu
            'max_depth': None,  # Cho phép cây phát triển tự do với dữ liệu đủ lớn
            'min_samples_split': 10,  # Tăng để tránh overfitting
            'min_samples_leaf': 4,  # Đảm bảo mỗi lá cây có đủ mẫu
            'max_features': 'sqrt',  
            'random_state': 42,
            'n_jobs': -1  
        }

    def read_data(self, path):
        """Read data from CSV file."""
        data = pd.read_csv(path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data.fillna(0, inplace=True)
        return data
    
    def prepare_sequences(self, data, prediction_step=50):
        """Prepare sequential input for time series prediction."""
        X, y = [], []
        for i in range(prediction_step, len(data)):
            X.append(data.iloc[i-prediction_step:i].values.flatten())
            y.append(data.iloc[i]['close'])
        return np.array(X), np.array(y)
    
    def split_and_transform_data(self, data, split_ratio=0.8):
        """Split data into train and test sets."""
        # Prepare sequential data
        X, y = self.prepare_sequences(data)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=split_ratio, shuffle=False
        )
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print('X_train shape:', X_train_scaled.shape)
        print('X_test shape:', X_test_scaled.shape)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)
        
        return data, X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_rf_model(self):
        """Create Random Forest Regressor model."""
        return RandomForestRegressor(**self.rf_params)
    
    def fit(self, rf_model, X_train, y_train):
        """Fit the Random Forest model."""
        rf_model.fit(X_train, y_train)
        return rf_model
    
    def save_model(self, rf_model, filename):
        """Save the trained Random Forest model to a file."""
        joblib.dump(rf_model, filename)
        print(f'Model saved to {filename}')

    def predict(self, rf_model, X_test):
        """Make predictions using the Random Forest model."""
        return rf_model.predict(X_test)
    
    def plot_result(self, data, X_test, y_test, rf_pred):
        """Plot actual vs predicted results."""
        # Determine the test data's index
        test_start = data.index[len(data) - len(y_test) - self.prediction_step]
        test_index = data.index[len(data) - len(y_test):]
        
        # Ensure rf_pred has the same length as y_test and test_index
        rf_pred = rf_pred[:len(y_test)]
        
        fig, axes = plt.subplots(2, 1, sharex=False, figsize=(9, 6), 
                                tight_layout=True, 
                                gridspec_kw={'height_ratios': [1.5, 1]})

        # Full data plot
        axes[0].plot(data['close'], label='Full Data')
        axes[0].plot(test_index, y_test, linewidth=1.5, label='Actual Test')
        axes[0].plot(test_index, rf_pred, color='green', label='Forecast')
        axes[0].set_title('Close Price Estimations: Random Forest')
        axes[0].legend()

        # Zoomed view
        axes[1].plot(test_index, y_test, linewidth=1.5, label='Actual Test')
        axes[1].plot(test_index, rf_pred, label='Forecast')
        axes[1].set_xlim(pd.Timestamp('2023-01-01'), data.index[-1])
        axes[1].set_ylim(min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max()))
        axes[1].legend()

        plt.show()
    
    def cross_val_rf(self, X, y, k=6):
        """Perform k-fold cross-validation."""
        rf_model = self.get_rf_model()
        kfold = KFold(n_splits=k, shuffle=False)
        
        # Compute RMSE for each fold
        cv_scores = cross_val_score(
            rf_model, X, y, 
            scoring='neg_root_mean_squared_error', 
            cv=kfold
        )
        
        # Convert to positive RMSE
        return -cv_scores
    
    def plot_feature_importance(self, rf_model, X):
        """Plot feature importances."""
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices])
        plt.show()
    
    def plot_residuals(self, y_true, y_pred):
        """Plot residuals distribution."""
        residuals = y_true - y_pred
        plt.figure(figsize=(6, 3))
        sns.histplot(residuals, kde=True, bins=50)
        plt.axvline(x=residuals.mean(), color='red')
        plt.title('Close Price Residuals: Random Forest')
        plt.xlabel('Residuals')
        plt.show()
    
    def baseline_rf(self, path):
        """Main method to run Random Forest regression."""
        # Load data
        data = self.read_data(path)
        
        # Prepare data
        full_data, X_train, X_test, y_train, y_test = self.split_and_transform_data(data)
        
        # Create and train model
        rf_model = self.get_rf_model()
        rf_model = self.fit(rf_model, X_train, y_train)
        self.save_model(rf_model, "../model_save/rf_model.pkl")
        # Predict
        rf_pred = self.predict(rf_model, X_test)
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        cv_rmse = np.mean(self.cross_val_rf(X_train, y_train))
        r2 = r2_score(y_test, rf_pred)
        mae = mean_absolute_error(y_test, rf_pred)
        mape = mean_absolute_percentage_error(y_test, rf_pred)
        
        # Print results
        print('Testing performance:')
        print('--------------------')
        print(f'RMSE: {rmse:.4f}')
        print(f'6-fold CV RMSE: {cv_rmse:.4f}')
        print(f'R2: {r2:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'MAPE: {mape * 100:.4f}%')
        
        # Save results
        results = {
            'Metric': ['RMSE', '6-fold CV', 'R2', 'MAE', 'MAPE'],
            'Value': [rmse, cv_rmse, r2, mae, mape]
        }
        df = pd.DataFrame(results)
        path = "../results/rf/evaluation.csv"
        df.to_csv(path, index=False)
        print(f'Results saved to {path}')
        
        # Visualizations
        self.plot_result(full_data, X_test, y_test, rf_pred)
        self.plot_feature_importance(rf_model, X_train)
        self.plot_residuals(y_test, rf_pred)

if __name__ == '__main__':
    print("-----------------RUN-----------------")
    path = '../data/data_train_model.csv'
    _s = RfConfig()
    _s.baseline_rf(path)
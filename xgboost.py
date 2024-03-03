import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Assuming you have the following variables defined
claims = np.array([7300, 7700, 8000, 5700, 7800, 7600, 8000, 6700, 6200, 6800, 6700, 5800, 6200, 5500])
time_step = np.arange(len(claims))
split_time = 10

time_train = time_step[:split_time]
x_train = claims[:split_time]

time_valid = time_step[split_time:]
x_valid = claims[split_time:]

# Use Min-Max scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
x_valid_scaled = scaler.transform(x_valid.reshape(-1, 1)).flatten()

# Function for creating a windowed dataset
def windowed_dataset(series, window_size):
    features = []
    labels = []
    for i in range(len(series) - window_size):
        window = series[i: i + window_size]
        target = series[i + window_size]
        features.append(window)
        labels.append(target)
    return np.array(features), np.array(labels)

# Set window size
window_size = 3  # Choose a larger window size

# Create windowed dataset for training
X_train_np, y_train_np = windowed_dataset(x_train_scaled, window_size)

# Create windowed dataset for validation
X_valid_np, y_valid_np = windowed_dataset(x_valid_scaled, window_size)

# Reshape features for XGBoost
X_train_np = X_train_np.reshape((X_train_np.shape[0], X_train_np.shape[1]))
X_valid_np = X_valid_np.reshape((X_valid_np.shape[0], X_valid_np.shape[1]))

# Create XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Use TimeSeriesSplit for time series data
tscv = TimeSeriesSplit(n_splits=5)

# Train the model
model.fit(X_train_np, y_train_np)

# Predictions on validation set
y_valid_pred = model.predict(X_valid_np)

# Inverse transform predictions to original scale
y_valid

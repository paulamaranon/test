####Autoregression (AR)####
'''
Autoregression (AR) is a time series forecasting method that models the relationship between an observation and several lagged observations (previous time steps). In other words, it uses the past values of the time series to predict future values. The basic idea is that the value at a given time step is a linear combination of its past values.
The number of lags, denoted by p, is a crucial parameter in autoregressive models. It determines how many previous time steps are considered when predicting the current time step.
'''
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler

# Assuming you have the following variables defined
claims = np.array([7300, 7700, 8000, 5700, 7800, 7600, 8000, 6700, 6200, 6800, 6700, 5800, 6200, 5500])
time_step = np.arange(len(claims))
split_time = 9

time_train = time_step[:split_time]
x_train = claims[:split_time]

time_valid = time_step[split_time:]
x_valid = claims[split_time:]

# Use Min-Max scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
x_valid_scaled = scaler.transform(x_valid.reshape(-1, 1)).flatten()


# Function for creating lag features
def create_lagged_features(series, lag):
    lagged_series = np.roll(series, lag)
    lagged_series[:lag] = 0  # Set the initial values to 0 (or any other suitable value)
    return lagged_series


# Set the number of lags for the autoregressive model
lags = 1

# Create lagged features for training and validation sets
x_train_lagged = create_lagged_features(x_train_scaled, lags)
x_valid_lagged = create_lagged_features(x_valid_scaled, lags)

# Fit an autoregressive model
model = AutoReg(x_train_scaled, lags=lags)
model_fit = model.fit()

# Plot the original data, training data, validation data, and predictions
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(2, 1, 1)
plt.plot(time_step, claims, label='Original Data', marker='o', linestyle='-', color='black')

# Plot training data (inverse transform x_train_scaled)
plt.plot(time_step[:split_time], scaler.inverse_transform(x_train_scaled.reshape(-1, 1)).flatten(),
         label='Training Data', color='blue')

# Plot validation data (inverse transform x_valid_scaled)
plt.plot(time_step[split_time:], scaler.inverse_transform(x_valid_scaled.reshape(-1, 1)).flatten(),
         label='Validation Data', color='red')

# Plot predictions on validation set for the entire duration
time_forecast_valid = np.arange(len(x_valid_scaled)) + split_time
forecast_valid_scaled = model_fit.predict(start=len(x_train_scaled),
                                          end=len(x_train_scaled) + len(x_valid_scaled) - 1).flatten()
forecast_valid = scaler.inverse_transform(forecast_valid_scaled.reshape(-1, 1)).flatten()
plt.plot(time_forecast_valid, forecast_valid, label='Predictions (1 step ahead)', linestyle='--', color='green')

# Extend predictions two time steps ahead
for i in range(2):
    # Predict one time step ahead
    forecast_future_scaled = model_fit.predict(start=len(x_train_scaled) + len(x_valid_scaled) + i,
                                               end=len(x_train_scaled) + len(x_valid_scaled) + i)
    forecast_future = scaler.inverse_transform(forecast_future_scaled.reshape(-1, 1)).flatten()

    # Update the time and forecast arrays
    time_forecast_valid = np.append(time_forecast_valid, time_forecast_valid[-1] + 1)

    # Plot the updated predictions
    plt.plot(time_forecast_valid, forecast_future, linestyle='--', color='green',
             alpha=0.5)  # Adjust alpha for transparency

plt.title('Original Data, Training Data, Validation Data, and Predictions')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

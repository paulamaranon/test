
import csv

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb

split_time = 19
time_step = []
JEWL = []

with open("claims.txt", "r", encoding="utf-16") as f:
        reader = csv.reader(f, delimiter=",")

        # Skip the header
        next(reader)
        # Iterate through non-null lines
        for line in reader:
            # Assuming the first column is the date and the second column is the number
            time_step.append(float(line[0]))
            JEWL.append(float(line[2]))

time_train = time_step[:split_time]
time_valid = time_step[split_time:]
x_train=JEWL[:split_time]
x_valid = JEWL[split_time:]

# Use Min-Max scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(np.array(x_train).reshape(-1, 1)).flatten()
x_valid_scaled = scaler.transform(np.array(x_valid).reshape(-1, 1)).flatten()

# Reshape the time series data for XGBoost
x_train_reshaped = np.array(x_train_scaled).reshape(-1, 1)
x_valid_reshaped = np.array(x_valid_scaled).reshape(-1, 1)

# Create an XGBoost model
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# Fit the model on the training data
xgboost_model.fit(np.arange(len(x_train_reshaped)).reshape(-1, 1), x_train_reshaped)

# Make predictions on the validation set
xgboost_predictions_scaled = xgboost_model.predict(np.arange(len(x_train_reshaped), len(x_train_reshaped) + len(x_valid_reshaped)).reshape(-1, 1))

# Denormalize the predictions
xgboost_predictions_denormalized = scaler.inverse_transform(xgboost_predictions_scaled.reshape(-1, 1)).flatten()

# Plot the original data, training data, validation data, and predictions
plt.figure(figsize=(10, 6))

# Plot original data in blue
plt.plot(time_step, JEWL, label='Original Data', color='blue')

# Plot training data in green
plt.plot(time_step[:split_time], JEWL[:split_time], label='Training Data', color='green')

# Plot validation data in orange
plt.plot(time_valid, JEWL[split_time:], label='Validation Data', color='orange')

# Plot predictions (Validation)
plt.plot(time_valid, xgboost_predictions_denormalized, label='Predictions (Validation)', color='red')

plt.legend()
plt.title('Original Data, Training Data, Validation Data, and Predictions (XGBoost)')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.show()

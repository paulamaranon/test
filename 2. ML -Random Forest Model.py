'''
The "Validation Data" portion of the plot shows the actual values from the original validation set (x_valid_scaled), and the "Predictions" line represents the model's predictions on the validation set, both aligned with the corresponding time steps. This allows you to visually assess how well the model's predictions match the actual validation data.
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
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
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten().reshape(-1, 1)
x_valid_scaled = scaler.transform(x_valid.reshape(-1, 1)).flatten().reshape(-1, 1)

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)  # Adjust hyperparameters

# Train the model
model.fit(time_train.reshape(-1, 1), x_train_scaled)

# Make predictions on validation set
forecast_valid_scaled = model.predict(time_valid.reshape(-1, 1))
forecast_valid_scaled = forecast_valid_scaled.reshape(-1, 1)  # Reshape to 2D array

# Flatten the predictions and inverse transform
forecast_valid = scaler.inverse_transform(forecast_valid_scaled.reshape(-1, 1)).flatten()

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(scaler.inverse_transform(x_valid_scaled), forecast_valid)
print("Mean Absolute Error (MAE):", mae)

# Plot the original data, training data, test data, and predictions
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(2, 1, 1)
plt.plot(time_step, claims, label='Original Data', marker='o', linestyle='-', color='black')

# Plot training data (inverse transform x_train_scaled)
plt.plot(time_step[:split_time], scaler.inverse_transform(x_train_scaled).flatten(), label='Training Data', color='blue')

# Plot validation data (inverse transform x_valid_scaled)
plt.plot(time_step[split_time:], scaler.inverse_transform(x_valid_scaled).flatten(), label='Validation Data', color='red')

# Plot predictions on validation set for the entire duration
forecast_valid_scaled = model.predict(time_valid.reshape(-1, 1))

# Flatten the predictions and inverse transform
forecast_valid = scaler.inverse_transform(forecast_valid_scaled.reshape(-1, 1)).flatten()

# Use the same indexing as for validation data
time_forecast_valid = time_step[split_time:split_time + len(forecast_valid)]

# Ensure dimensions match
assert len(time_forecast_valid) == len(forecast_valid), "Dimensions do not match"

# Plot predictions on validation set along the validation time steps
plt.plot(time_valid, forecast_valid, label='Predictions', linestyle='--', color='green')


# Extend predictions one month ahead
for i in range(3):  # Adjust the number of time steps into the future
    # Predict one time step ahead
    next_time_step = time_valid[-1] + 1
    forecast_future_scaled = model.predict([[next_time_step]])
    forecast_future = scaler.inverse_transform(forecast_future_scaled.reshape(-1, 1)).flatten()

    # Update the time and forecast arrays
    time_forecast_valid = np.append(time_forecast_valid, next_time_step)
    forecast_valid = np.append(forecast_valid, forecast_future)

    # Plot the updated predictions
    plt.plot(time_forecast_valid, forecast_valid, linestyle='--', color='green', alpha=0.5)  # Adjust alpha for transparency

    # Update the input sequence for the next prediction
    time_valid = np.append(time_valid, next_time_step)

plt.title('Original Data, Training Data, Validation Data, and Predictions')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

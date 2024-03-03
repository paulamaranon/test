import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Read data from CSV file
df = pd.read_csv(r"claims.txt", parse_dates=["Date"], index_col=["Date"], encoding="utf-16")
print(df.head())

# Extract tickets and time_step
tickets = df["Number"].to_numpy()
time_step = np.arange(len(tickets))

# Split the data into training and validation sets
split_time = 15
time_train = time_step[:split_time]
x_train = tickets[:split_time]

time_valid = time_step[split_time:]
x_valid = tickets[split_time:]


# Use Min-Max scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten().reshape(-1, 1)
x_valid_scaled = scaler.transform(x_valid.reshape(-1, 1)).flatten().reshape(-1, 1)

# Naive forecast
naive_forecast = np.roll(x_valid_scaled, 1) #creates a naive forecast by shifting the values in the x_valid_scaled array by one position to the right
naive_forecast[0] = x_train_scaled[-1]  # Set the first value of the naive forecast to the last value of the training set

# Calculate Mean Absolute Error (MAE) for the naive forecast
mae_naive = mean_absolute_error(x_valid_scaled, naive_forecast)
print("Mean Absolute Error (Naive Forecast):", mae_naive)

# Print values for the last time step
last_time_step = min(time_step[-1], len(x_valid) - 1)
print(f'Ground truth at time step {last_time_step}: {x_valid[last_time_step]}')
if last_time_step + 1 < len(x_valid):
    print(f'Naive forecast at time step {last_time_step + 1}: {scaler.inverse_transform(naive_forecast)[last_time_step + 1]}')
else:
    print(f'Naive forecast at time step {last_time_step + 1}: (Not available, validation set size exceeded)')


# Plot the original data, training data, validation data, and naive forecast
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(2, 1, 1)
plt.plot(time_step, tickets, label='Original Data', marker='o', linestyle='-', color='black')
# Plot training data
plt.plot(time_step[:split_time], x_train, label='Training Data', color='blue')
# Plot validation data
plt.plot(time_step[split_time:], x_valid, label='Validation Data', color='red')
# Plot naive forecast
plt.plot(time_step[split_time:], scaler.inverse_transform(naive_forecast).flatten(), label='Naive Forecast', linestyle='--', color='green')

plt.title('Original Data, Training Data, Validation Data, and Naive Forecast')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

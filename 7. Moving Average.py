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
split_time = 20
time_train = time_step[:split_time]
x_train = tickets[:split_time]

time_valid = time_step[split_time:]
x_valid = tickets[split_time:]


# Use Min-Max scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten().reshape(-1, 1)
x_valid_scaled = scaler.transform(x_valid.reshape(-1, 1)).flatten().reshape(-1, 1)

# MA
#This function generate forecasts for the entire length of the series, including the validation period.
def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series)):
        if time < window_size:
            forecast.append(np.nan)
        else:
            forecast.append(series[time - window_size:time].mean())
    return np.array(forecast)


# Adjust the window size based on the available training data
window_size = 12

# Generate the moving average forecast
moving_avg = moving_average_forecast(tickets, window_size)

# Extract the relevant portion for the validation set
moving_avg_validation = moving_avg[split_time: split_time + len(x_valid)]
# Plot the original data, training data, validation data, and naive forecast
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(2, 1, 1)
plt.plot(time_step, tickets, label='Original Data', marker='o', linestyle='-', color='black')

# Plot training data
plt.plot(time_step[:split_time], x_train, label='Training Data', color='blue')

# Plot validation data
plt.plot(time_step[split_time:], x_valid, label='Validation Data', color='red')

# Plot MA forecast
plt.plot(time_step[split_time: split_time + len(x_valid)], moving_avg_validation, label='Moving Average', linestyle='--', color='green')

plt.title('Original Data, Training Data, Validation Data, and Naive Forecast')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Import with pandas
import pandas as pd
import numpy as np
# Parse dates and set date column to index
df = pd.read_csv(r"C:\Users\paula\Documents\UK LAPTOP\17 GITHUB\workspace\timeseries\claims.txt",
                 parse_dates=["Date"],
                 index_col=["Date"], encoding="utf-16") # parse the date column (tell pandas column 1 is a datetime)
print(df.head())
# How many samples do we have?
print(len(df))

import matplotlib.pyplot as plt
df.plot(figsize=(10, 7))
plt.ylabel("Ticket")
plt.title("Tickets per month", fontsize=16)
plt.legend(fontsize=14)
plt.show()

# Get bitcoin date array
timesteps = df.index.to_numpy()
tickets = df["Number"].to_numpy()

print(timesteps[:10], tickets[:10])

#Create train & test sets for time series (the right way
# Create train and test splits the right way for time series data
split_size = int(0.8 * len(tickets)) # 80% train, 20% test

# Create train data splits (everything before the split)
X_train, y_train = timesteps[:split_size], tickets[:split_size]

# Create test data splits (everything after the split)
X_test, y_test = timesteps[split_size:], tickets[split_size:]

print(f"X_train: {len(X_train)}, X_test: {len(X_test)}, y_train: {len(y_train)}, y_test: {len(y_test)}")



# Create a function to plot time series data
def plot_time_series(timesteps, values, color='blue', format='.', start=0, end=None, label=None):
    plt.plot(timesteps[start:end], values[start:end], format, color=color, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)  # make label bigger
    plt.grid(True)

# Concatenate train and test data
combined_timesteps = np.concatenate([X_train, X_test])
combined_values = np.concatenate([y_train, y_test])

# Plot combined data
# Plot combined data
plt.figure(figsize=(10, 7))
plot_time_series(timesteps=combined_timesteps, values=combined_values, label="Train data", color='blue')
plot_time_series(timesteps=X_test, values=y_test, label="Test data", color='red')

plt.show()

#horizon = number of timesteps to predict into future
#window = number of timesteps from past used to predict horizon

#Model 0: Naïve forecast (baseline)
# Create a naïve forecast
naive_forecast = y_test[:-1] # Naïve forecast equals every value excluding the last value
print(f"Navive forecast first 10:{naive_forecast[:10]}, naive forecast last 10: {naive_forecast[-10:]}") # View frist 10 and last 10

# Plot naive forecast
# Plot naive forecast
plt.figure(figsize=(10, 7))
plot_time_series(timesteps=X_train, values=y_train, label="Train data")
plot_time_series(timesteps=X_test, values=y_test, label="Test data", color='blue')
plot_time_series(timesteps=X_test[1:], values=naive_forecast, format="-", label="Naive forecast", color='orange')

plt.show()

plt.figure(figsize=(10, 7))
offset = 300 # offset the values by 300 timesteps
plot_time_series(timesteps=X_test, values=y_test, start=offset, label="Test data")
plot_time_series(timesteps=X_test[1:], values=naive_forecast, format="-", start=offset, label="Naive forecast")
plt.show()
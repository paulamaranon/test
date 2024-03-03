'''
Linear regression models fit a straight line to the data, and the slope of the line is determined by the training data. In your case, if the first two values in the validation data have a certain trend, the linear regression model will extend that trend in a straight line.
So, if the initial part of your validation data shows an increasing or decreasing pattern, the linear regression model will project that trend into the future. Linear regression is simple and assumes a linear relationship between the input features and the target variable, which might not capture more complex patterns in the data. If your data has nonlinear patterns, you might want to explore more complex models like polynomial regression, decision trees, or other machine learning models that can capture nonlinear relationships.

The concept of accuracy is typically associated with classification problems rather than regression problems. In regression, accuracy is not the appropriate metric because it measures how well a model correctly predicts discrete classes, which is not applicable to continuous target values.

In the context of linear regression, it's more common to use metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE) to evaluate the performance of the model. These metrics quantify the difference between the predicted values and the actual values.
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
# Parse dates and set date column to index
df = pd.read_csv(r"C:\Users\paula\Documents\UK LAPTOP\17 GITHUB\workspace\timeseries\claims.txt",
                 parse_dates=["Date"],
                 index_col=["Date"], encoding="utf-16") # parse the date column (tell pandas column 1 is a datetime)
print(df.head())
tickets = df["Number"].to_numpy()
time_step = np.arange(len(tickets))
split_time = 15

time_train = time_step[:split_time]
x_train = tickets[:split_time]

time_valid = time_step[split_time:]
x_valid = tickets[split_time:]

# Use Min-Max scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten().reshape(-1, 1)
x_valid_scaled = scaler.transform(x_valid.reshape(-1, 1)).flatten().reshape(-1, 1)

model = LinearRegression()

# Train the model
model.fit(time_train.reshape(-1, 1), x_train_scaled)

# Plot the original data, training data, validation data, and predictions
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(2, 1, 1)
plt.plot(time_step, tickets, label='Original Data', marker='o', linestyle='-', color='black')

# Plot training data (inverse transform x_train_scaled)
plt.plot(time_step[:split_time], scaler.inverse_transform(x_train_scaled).flatten(), label='Training Data', color='blue')

# Plot validation data (inverse transform x_valid_scaled)
plt.plot(time_step[split_time:], scaler.inverse_transform(x_valid_scaled).flatten(), label='Validation Data', color='red')

# Plot predictions on validation set for the entire duration
forecast_valid_scaled = model.predict(time_valid.reshape(-1, 1))

# Flatten the predictions and inverse transform
forecast_valid = scaler.inverse_transform(forecast_valid_scaled).flatten()

# Calculate Mean Absolute Error (MAE)
#calculates the MAE by comparing the original (inverse-transformed) values of the validation set (x_valid_scaled) with the predicted values (forecast_valid).
mae = mean_absolute_error(scaler.inverse_transform(x_valid_scaled), forecast_valid)
print("Mean Absolute Error (MAE):", mae)

# Use the same indexing as for validation data
time_forecast_valid = time_step[split_time:split_time + len(forecast_valid)]

# Ensure dimensions match
assert len(time_forecast_valid) == len(forecast_valid), "Dimensions do not match"

# Plot predictions on validation set along the validation time steps
plt.plot(time_valid, forecast_valid, label='Predictions', linestyle='--', color='green')

# Extend predictions one time step ahead
for i in range(2):  # Adjust the number of time steps into the future
    # Predict one time step ahead
    next_time_step = time_valid[-1] + 1
    forecast_future_scaled = model.predict([[next_time_step]])
    forecast_future = scaler.inverse_transform(forecast_future_scaled).flatten()

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

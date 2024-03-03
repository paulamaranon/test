#https://neptune.ai/blog/random-forest-regression-when-does-it-fail-and-why
#The Random Forest Regressor is unable to discover trends that would enable it in extrapolating values that fall outside the training set.
'''
To implement the solution involving a combination of Lasso regression and Random Forest:
- Run Lasso regression on the training data to predict the target variable.
-Train a Random Forest on the residuals from the Lasso regression.
This code includes both Lasso regression and Random Forest, with the Random Forest trained on the residuals. This combination may help improve the predictions in cases where Random Forest alone struggles with extrapolation.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
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

# Step 1: Run Lasso regression
lasso_model = Lasso(alpha=0.1)  # Adjust alpha as needed
lasso_model.fit(time_train.reshape(-1, 1), x_train_scaled)
lasso_predictions_scaled = lasso_model.predict(time_valid.reshape(-1, 1))
lasso_predictions_scaled = lasso_predictions_scaled.reshape(-1, 1)

# Step 2: Train Random Forest on residuals
residuals = x_valid_scaled - lasso_predictions_scaled

# Use Random Forest on residuals
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(time_valid.reshape(-1, 1), residuals)

# Make predictions on validation set
forecast_valid_residuals = rf_model.predict(time_valid.reshape(-1, 1))
forecast_valid_residuals = forecast_valid_residuals.reshape(-1, 1)

# Combine Lasso predictions and Random Forest predictions
forecast_valid_scaled = lasso_predictions_scaled + forecast_valid_residuals

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

# Plot predictions on validation set along the validation time steps
plt.plot(time_valid, forecast_valid, label='Predictions', linestyle='--', color='green')

# Extend predictions one month ahead
for i in range(3):  # Adjust the number of time steps into the future
    # Predict one time step ahead
    next_time_step = time_valid[-1] + 1

    # Step 1: Lasso prediction
    lasso_forecast_scaled = lasso_model.predict([[next_time_step]])
    lasso_forecast_scaled = lasso_forecast_scaled.reshape(-1, 1)

    # Step 2: Random Forest prediction on residuals
    rf_forecast_residuals = rf_model.predict([[next_time_step]])
    rf_forecast_residuals = rf_forecast_residuals.reshape(-1, 1)

    # Combine Lasso and Random Forest predictions
    forecast_future_scaled = lasso_forecast_scaled + rf_forecast_residuals

    # Update the time and forecast arrays
    time_valid = np.append(time_valid, next_time_step)
    forecast_valid = np.append(forecast_valid, scaler.inverse_transform(forecast_future_scaled).flatten())

# Plot the updated predictions
plt.plot(time_valid, forecast_valid, linestyle='--', color='green', alpha=0.5)  # Adjust alpha for transparency

plt.title('Original Data, Training Data, Validation Data, and Predictions')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

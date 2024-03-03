import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

''
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

df = pd.DataFrame({'ds':time_step, 'y':JEWL})

# Load your time series data
# For this example, let's assume you have a DataFrame with a column 'value' representing your time series
# Replace this with your actual time series data
# df = pd.read_csv('your_data.csv')

# Assuming 'y' is your time series column
time_series = df['y']

# Plot the original time series
plt.plot(time_series, label='Original Time Series')
plt.title('Original Time Series')
plt.show()

# Fit ARIMA model
# You need to choose appropriate values for p, d, and q based on your data characteristics
p = 2  # Autoregressive order
d = 1  # Differencing order
q = 2  # Moving average order

# Create ARIMA model
arima_model = ARIMA(time_series, order=(p, d, q))

# Fit the model
arima_result = arima_model.fit()

# Make predictions
forecast_steps = 10  # Adjust as needed
arima_forecast = arima_result.get_forecast(steps=forecast_steps)

# Extract forecasted values and confidence intervals
forecast_values = arima_forecast.predicted_mean
conf_int = arima_forecast.conf_int()

# Plot the original time series and forecast
plt.plot(time_series, label='Original Time Series')
plt.plot(forecast_values, label='ARIMA Forecast', color='red')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()
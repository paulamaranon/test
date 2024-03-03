import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# Parse dates and set date column to index
df = pd.read_csv(r"C:\Users\paula\Documents\UK LAPTOP\17 GITHUB\workspace\timeseries\claims.txt",
                 parse_dates=["Date"],
                 index_col=["Date"], encoding="utf-16") # parse the date column (tell pandas column 1 is a datetime)
print(df.head())
# Assuming you have a DataFrame 'df' with columns 'Date' and 'Value'
# and 'Date' is in datetime64 format

# Extract features and target variable
X = df['Date'].values.reshape(-1, 1)  # Feature: Date
y = df['Value'].values  # Target: Value

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Plot the results
plt.scatter(X_test, y_test, label='Actual Data', color='blue')
plt.plot(X_test, y_pred, label='Linear Regression Prediction', color='orange')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load historical data
data = {
    'Date': ['2022-1-31', '2022-2-28', '2022-3-31', '2022-4-30', '2022-5-31', '2022-6-30', '2022-7-31', '2022-8-30', '2022-9-30', '2022-10-31', '2022-11-30', '2022-12-31', '2023-1-31', '2023-2-28', '2023-3-31', '2023-4-30', '2023-5-31', '2023-6-30', '2023-7-31', '2023-8-30', '2023-9-30', '2023-10-31', '2023-11-30'],
    'Number': [15727, 15382, 16436, 15529, 15978, 15640, 15905, 16012, 17339, 17724, 18002, 15758, 17845, 17614, 18022, 16713, 16269, 16825, 16795, 15858, 15879, 16287, 15599]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# Use Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Number'].values.reshape(-1, 1)).flatten()

# Create a windowed dataset
window_size = 3  # You may need to adjust this based on your model's window size
windowed_data = np.array([scaled_data[i:i+window_size] for i in range(len(scaled_data)-window_size)])
target_data = scaled_data[window_size:]

# Reshape the input data to match the model's input shape
X = windowed_data.reshape(-1, window_size, 1)
y = target_data.reshape(-1, 1)

# Build the model
model = Sequential([
    LSTM(100, activation='relu', return_sequences=True, input_shape=(window_size, 1)),
    Dropout(0.3),
    LSTM(100, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(100, activation='relu'),
    Dense(1)
])


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X, y, epochs=100, batch_size=1, validation_split=0.1, verbose=1, shuffle=False)


# Make predictions on the validation data
validation_predictions = []
for i in range(window_size, len(scaled_data)):
    input_data = np.array([scaled_data[i-window_size:i]]).reshape(1, window_size, 1)
    predicted_value_scaled = model.predict(input_data)[0, 0]
    validation_predictions.append(predicted_value_scaled)

# Denormalize the predictions
denormalized_predictions = scaler.inverse_transform(np.array(validation_predictions).reshape(-1, 1)).flatten()

# Print the actual and predicted values on the validation data
for i in range(len(denormalized_predictions)):
    actual_value = df['Number'][i + window_size]
    print(f"Actual: {actual_value}, Predicted: {denormalized_predictions[i]}")
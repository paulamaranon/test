import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Assuming you have the following variables defined
claims = np.array([7300, 7700, 8000, 5700, 7800, 7600, 8000, 6700, 6200, 6800, 6700, 5800, 6200, 5500])
time_step = np.arange(len(claims))
split_time = 8

time_train = time_step[:split_time]
x_train = claims[:split_time]

time_valid = time_step[split_time:]
x_valid = claims[split_time:]

# Use Min-Max scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
x_valid_scaled = scaler.transform(x_valid.reshape(-1, 1)).flatten()

    # Function for creating a windowed dataset
def windowed_dataset(series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    return dataset.batch(1).prefetch(1)

    # Set window size
window_size = 3

    # Create windowed dataset for training
dataset_train = windowed_dataset(x_train_scaled, window_size)

    # Create windowed dataset for validation
dataset_valid = windowed_dataset(x_valid_scaled, window_size)

    # Build a more complex LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

    # Compile the model
model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    # Train the model
history = model.fit(dataset_train, epochs=50, validation_data=dataset_valid)
    # Plot the original data, training data, test data, and predictions
plt.figure(figsize=(12, 6))

    # Plot original data
plt.subplot(2, 1, 1)
plt.plot(time_step, claims, label='Original Data', marker='o', linestyle='-', color='black')

    # Plot training data (inverse transform x_train_scaled)
plt.scatter(time_train[window_size:], scaler.inverse_transform(x_train_scaled[window_size:].reshape(-1, 1)).flatten(), label='Training Data', color='blue')

    # Plot validation data (inverse transform x_valid_scaled)
plt.scatter(time_valid[window_size:],scaler.inverse_transform(x_valid_scaled[window_size:].reshape(-1, 1)).flatten(),label='Validation Data', color='red')

    # Plot predictions on validation set
forecast_valid_scaled = model.predict(dataset_valid)
forecast_valid = scaler.inverse_transform(forecast_valid_scaled.flatten().reshape(-1, 1)).flatten()
    # Use the same indexing as for training and validation data
    # Plot predictions on validation set
plt.plot(time_valid[window_size:window_size + len(forecast_valid)], forecast_valid, label='Predictions', linestyle='--', color='green')

plt.title('Original Data, Training Data, Validation Data, and Predictions')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

    # Plot validation loss vs training loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Read data from CSV file
df = pd.read_csv(r"claims.txt", parse_dates=["Date"], index_col=["Date"], encoding="utf-16")
print(df.head())

# Extract tickets and time_step
tickets = df["Number"].to_numpy()
time_step = np.arange(len(tickets))
split_time = 20

time_train = time_step[:split_time]
x_train = tickets[:split_time]

time_valid = time_step[split_time:]
x_valid = tickets[split_time:]

# Use Min-Max scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
x_valid_scaled = scaler.transform(x_valid.reshape(-1, 1)).flatten()

# Function for creating a windowed dataset
def windowed_dataset(series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    return dataset.batch(1).prefetch(1)

# Set window size
window_size = 2  # Choose a larger window size

# Create windowed dataset for training
dataset_train = windowed_dataset(x_train_scaled, window_size)

# Create windowed dataset for validation
dataset_valid = windowed_dataset(x_valid_scaled, window_size)

# Build the N-Beats model
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, units, theta_size, backcast_size, forecast_size, input_shape, **kwargs):
        super(NBeatsBlock, self).__init__(**kwargs)
        self.units = units
        self.theta_size = theta_size
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.backcast_fc = tf.keras.layers.Dense(backcast_size, input_shape=input_shape)
        self.forecast_fc = tf.keras.layers.Dense(forecast_size, input_shape=input_shape)
        self.theta_fc = tf.keras.layers.Dense(theta_size, activation='relu', input_shape=input_shape)

    def call(self, inputs):
        x = inputs
        backcast = self.backcast_fc(x)
        forecast = self.forecast_fc(x)
        theta = self.theta_fc(backcast)
        return backcast, forecast, theta

class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, units, theta_size, backcast_size, forecast_size, input_shape, **kwargs):
        super(NBeatsBlock, self).__init__(**kwargs)
        self.units = units
        self.theta_size = theta_size
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.backcast_fc = tf.keras.layers.Dense(backcast_size)
        self.forecast_fc = tf.keras.layers.Dense(forecast_size)
        self.theta_fc = tf.keras.layers.Dense(theta_size, activation='relu')

    def call(self, inputs):
        x = inputs
        backcast = self.backcast_fc(x)
        forecast = self.forecast_fc(x)
        theta = self.theta_fc(backcast)
        return backcast, forecast, theta

class NBeats(tf.keras.models.Model):
    def __init__(self, stack_types, units, theta_size, backcast_size, forecast_size, stacks, input_shape, **kwargs):
        super(NBeats, self).__init__(**kwargs)
        self.stack_types = stack_types
        self.units = units
        self.theta_size = theta_size
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.stacks = stacks
        self.blocks = []
        for stack_type, unit in zip(stack_types, units):
            for _ in range(stacks):
                self.blocks.append(NBeatsBlock(unit, theta_size, backcast_size, forecast_size, input_shape))
                if stack_type == 'generic':
                    backcast_size += unit
                elif stack_type == 'trend':
                    backcast_size += 2 * unit
                elif stack_type == 'seasonality':
                    backcast_size += 4 * unit

    def call(self, inputs):
        x = inputs
        forecast = 0
        for block in self.blocks:
            backcast, block_forecast, theta = block(x)
            forecast += block_forecast
            x = tf.concat([x, backcast], axis=-1)
        return forecast


# Create the N-Beats model
nbeats_model = NBeats(stack_types=['generic'], units=[128], theta_size=1,
                      backcast_size=window_size, forecast_size=1, stacks=3, input_shape=(window_size,))

# Compile the model
nbeats_model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(), metrics=['mae'])

# Train the N-Beats model
nbeats_history = nbeats_model.fit(dataset_train, epochs=100)

# Plot the original data, training data, test data, and N-Beats predictions
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(2, 1, 1)
plt.plot(time_step, tickets, label='Original Data', marker='o', linestyle='-', color='black')

# Plot training data (inverse transform x_train_scaled)
plt.plot(time_step[:split_time], scaler.inverse_transform(x_train_scaled.reshape(-1, 1)).flatten(), label='Training Data', color='blue')

# Plot validation data (inverse transform x_valid_scaled)
plt.plot(time_step[split_time:], scaler.inverse_transform(x_valid_scaled.reshape(-1, 1)).flatten(), label='Validation Data', color='red')

# Plot predictions on validation set for the entire duration
forecast_valid_scaled = nbeats_model.predict(dataset_valid)

# Flatten the predictions and inverse transform
forecast_valid = scaler.inverse_transform(forecast_valid_scaled.flatten().reshape(-1, 1)).flatten()

# Use the same indexing as for validation data
time_forecast_valid = time_step[split_time:split_time + len(forecast_valid)]

# Ensure dimensions match
assert len(time_forecast_valid) == len(forecast_valid), "Dimensions do not match"

# Plot predictions on validation set along the validation time steps
plt.plot(time_valid[:len(forecast_valid)], forecast_valid, label='N-Beats Predictions', linestyle='--', color='green')

# Extend predictions one month ahead
for i in range(3):  # Adjust the number time steps into the future
    # Predict one time step ahead
    forecast_future_scaled = nbeats_model.predict(np.array([x_valid_scaled[-window_size:]]))
    forecast_future = scaler.inverse_transform(forecast_future_scaled.flatten().reshape(-1, 1)).flatten()

    # Update the time and forecast arrays
    time_forecast_valid = np.append(time_forecast_valid, time_forecast_valid[-1] + 1)
    forecast_valid = np.append(forecast_valid, forecast_future)

    # Plot the updated predictions
    plt.plot(time_forecast_valid, forecast_valid, linestyle='--', color='green', alpha=0.5)  # Adjust alpha for transparency

    # Update the input sequence for the next prediction
    x_valid_scaled = np.append(x_valid_scaled, forecast_future[-1])

plt.title('Original Data, Training Data, Validation Data, and N-Beats Predictions')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

# Plot N-Beats training loss
plt.subplot(2, 1, 2)
plt.plot(nbeats_history.history['loss'], label='N-Beats Training Loss', color='purple')
plt.title('N-Beats Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

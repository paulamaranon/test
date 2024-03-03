import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def plot_series(time, series, format="-", start=0, end=None):
    """
    Visualizes time series data

    Args:
      time (array of int) - contains the time steps
      series (array of int) - contains the measurements for each time step
      format - line style when plotting the graph
      label - tag for the line
      start - first time step to plot
      end - last time step to plot
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))

    if type(series) is tuple:

        for series_num in series:
            # Plot the time series data
            plt.plot(time[start:end], series_num[start:end], format)

    else:
        # Plot the time series data
        plt.plot(time[start:end], series[start:end], format)

    # Label the x-axis
    plt.xlabel("Time")

    # Label the y-axis
    plt.ylabel("Value")

    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()

# Parameters
series = np.array([7300, 7700, 8000, 5700, 7800, 7600, 8000, 6700, 6200, 6800, 6700, 5800, 6200, 5500])
time = np.arange(len(series))

# Plot the results
plot_series(time, series)

# Split the Dataset
# Define the split time
split_time = 11

# Get the train set
time_train = time[:split_time]
x_train = series[:split_time]

# Get the validation set
time_valid = time[split_time:]
x_valid = series[split_time:]

# Normalize the data
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
x_valid_scaled = scaler.transform(x_valid.reshape(-1, 1)).flatten()

# Plot the train set
plot_series(time_train, x_train)
# Plot the validation set
plot_series(time_valid, x_valid)

# Prepare features and labels
# Parameters
window_size = 2
batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

# Generate the dataset windows
# Create windowed dataset for training
dataset_train = windowed_dataset(x_train_scaled, window_size, batch_size,shuffle_buffer_size)
# Create windowed dataset for validation
dataset_valid = windowed_dataset(x_valid_scaled, window_size,batch_size,shuffle_buffer_size)

# Build and compile the model
# Build the single layer neural network
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])

# Print the initial layer weights
print("Layer weights: \n {} \n".format(l0.get_weights()))

# Print the model summary
model.summary()

# Learning rate
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
# Set the training parameters
model.compile(loss="mse", optimizer=optimizer)

# Train the model
history = model.fit(dataset_train, epochs=30)

# Print the layer weights
print("Layer weights {}".format(l0.get_weights()))

# Model Prediction
# Generate a model prediction for the validation set
# Model Prediction
# Generate a model prediction for the validation set
forecast = []

# Use the model to predict data points one step at a time
for time in range(len(series) - window_size):
    # Extract the window of data for prediction
    input_window = x_valid_scaled[time:time + window_size]

    # Ensure the correct input shape for the model
    input_window = np.reshape(input_window, (1, window_size))  # Ensure the correct shape

    # Predict one time step ahead
    prediction = model.predict(input_window)

    # Append the prediction to the forecast list
    forecast.append(prediction)

# Slice the points that are aligned with the validation set
forecast = np.array(forecast).squeeze()

# Reverse the normalization for forecast
forecast_denormalized = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

# Denormalize the validation set
x_valid_denormalized = scaler.inverse_transform(x_valid_scaled.reshape(-1, 1)).flatten()

# Overlay the results with the validation set
plot_series(time_valid, (x_valid_denormalized, forecast_denormalized))

# Compute the metrics
mae = tf.keras.metrics.mean_absolute_error(x_valid_denormalized, forecast_denormalized).numpy()
print("MAE:", mae)

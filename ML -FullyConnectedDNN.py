import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.src.optimizers.optimizer_v1 import adam

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


def trend(time, slope=0):
    """
    Generates synthetic data that follows a straight line given a slope value.

    Args:
      time (array of int) - contains the time steps
      slope (float) - determines the direction and steepness of the line

    Returns:
      series (array of float) - measurements that follow a straight line
    """

    # Compute the linear series given the slope
    series = slope * time

    return series

# Parameters
series = series = np.array([7300, 7700, 8000, 5700, 7800, 7600, 8000, 6700, 6200, 6800, 6700, 5800, 6200, 5500])
time = np.arange(len(series))


# Plot the results
plot_series(time, series)
#Split the Dataset
# Define the split time
split_time = 10

# Get the train set
time_train = time[:split_time]
x_train = series[:split_time]

# Check for NaN or Inf values in the original data
if np.any(np.isnan(x_train)) or np.any(np.isinf(x_train)):
    raise ValueError("Input data contains NaN or Inf values.")

# Normalize the data
mean = np.mean(x_train)
std = np.std(x_train)

# Check for division by zero
if std == 0:
    raise ValueError("Standard deviation is zero. Adjust the data or use a small constant.")

x_train_normalized = (x_train - mean) / std

# Get the validation set
time_valid = time[split_time:]
x_valid = series[split_time:]

# Normalize the validation data
x_valid_normalized = (x_valid - mean) / std
# Check for NaN or Inf values in the normalized data
if np.any(np.isnan(x_train_normalized)) or np.any(np.isinf(x_train_normalized)):
    raise ValueError("Normalized data contains NaN or Inf values.")
# Plot the train set
plot_series(time_train, x_train)
# Plot the validation set
plot_series(time_valid, x_valid)

#########Prepare features and labels
# Parameters
window_size = 2
batch_size = 32
shuffle_buffer_size = 1000
#One thing to note here is the window_size + 1 when you call dataset.window(). There is a + 1 to indicate that you're taking the next point as the label. For example, the first 20 points will be the feature so the 21st point will be the label.

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
train_dataset = windowed_dataset(x_train_normalized, window_size, batch_size, shuffle_buffer_size)
##########Build and compile the model
# Build the single layer neural network

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60,input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Print the model summary
model.summary()
# Set the training parameters
#model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-9, momentum=0.9))
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
# Train the model
model.fit(train_dataset,epochs=20)


##############Model Prediction
#generate a model prediction by passing a batch of data windows. If you will be slicing a window from the original series array, you will need to add a batch dimension before passing it to the model. That can be done by indexing with the np.newaxis constant or using the np.expand_dims() method.
# Initialize a list
forecast = []

# Use the model to predict data points per window size
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

# Slice the points that are aligned with the validation set
forecast = forecast[split_time - window_size:]

# Compare number of elements in the predictions and the validation set
print(f'length of the forecast list: {len(forecast)}')
print(f'shape of the validation set: {x_valid.shape}')

# Preview shapes after using the conversion and squeeze methods
print(f'shape after converting to numpy array: {np.array(forecast).shape}')
print(f'shape after squeezing: {np.array(forecast).squeeze().shape}')

# Convert to a numpy array and drop single dimensional axes
results = np.array(forecast).squeeze()

# Overlay the results with the validation set
plot_series(time_valid, (x_valid, results))

# Compute the metrics
print(tf.keras.metrics.mean_squared_error(x_valid, results).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
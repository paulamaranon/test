import csv

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from tensorflow.python.keras.layers import LSTM

'''
# Function to perform linear extrapolation
def linear_extrapolation(time, values):
    interp_func = interp1d(time, values, kind='linear', fill_value='extrapolate')
    new_time = np.linspace(time[0], time[-1], num_extrapolation_points)
    new_values = interp_func(new_time)
    return new_time, new_values
'''

# Assuming you have the following variables defined
dummy_data = np.array([70, 60, 65, 80, 70, 65,73, 68, 65, 69, 68, 65, 70, 68, 65,70, 68, 65,70, 68, 65, 70, 60, 65, 80, 70, 65,73, 68, 65, 69, 68, 65, 70, 68, 65,70, 68, 65,70, 68, 65])
time_step = np.arange(len(dummy_data))
split_time = 19 
'''
# Set the number of extrapolation points
num_extrapolation_points = 30

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

'''

time_train = time_step[:split_time]
time_valid = time_step[split_time:]
x_train=dummy_data[:split_time]
x_valid = dummy_data[split_time:]

# Use Min-Max scaling
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(np.array(x_train).reshape(-1, 1)).flatten()
x_valid_scaled = scaler.transform(np.array(x_valid).reshape(-1, 1)).flatten()

# Perform linear extrapolation on the entire dataset
#extrapolated_time, extrapolated_values = linear_extrapolation(time_step, x_scaled)

# Concatenate the original and extrapolated data
#combined_time = np.concatenate((time_step, extrapolated_time))
#combined_values = np.concatenate((x_scaled, extrapolated_values))

# Split the data into training and validation sets after data augmentation
#time_train = combined_time[:split_time]
#x_train_scaled = combined_values[:split_time]

#time_valid = combined_time[split_time:]
#x_valid_scaled = combined_values[split_time:]

# Function for creating a windowed dataset
def windowed_dataset(series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    return dataset.batch(1).prefetch(1)

# Set window size
window_size = 2

# Create windowed dataset for training
dataset_train = windowed_dataset(x_train_scaled, window_size)

# Create windowed dataset for validation
dataset_valid = windowed_dataset(x_valid_scaled, window_size)

# Build a more complex LSTM model
# Build a more complex LSTM model

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=128, kernel_size=3,
                      strides=1,
                      activation="relu",
                      padding='causal',
                      input_shape=[window_size, 1]),
  tf.keras.layers.LSTM(96, return_sequences=True),
  tf.keras.layers.LSTM(96),
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100)
])
'''
##########TUNE OPTIMAR NUMBER OF UNITS
# the unit_range will generate the following sequence of values: 32, 64, 96, 128, ..., 480, 512.
unit_range = range(32, 512, 32)
best_mae = float('inf')
best_num_units = 0

for num_units in unit_range:
    # Define the model
    model = tf.keras.models.Sequential([
      tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                         input_shape=[window_size]),
      tf.keras.layers.SimpleRNN(40, return_sequences=True),
      tf.keras.layers.SimpleRNN(40),
      tf.keras.layers.Dense(1),
      tf.keras.layers.Lambda(lambda x: x * 100.0)
    ])

    # Compile and train the model
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(lr=1e-07, momentum=0.9),
                  metrics=["mae"])
    history = model.fit(dataset_train, epochs=60, validation_data=dataset_valid)
    val_mae = history.history['val_mae'][-1]

    if val_mae < best_mae:
        best_mae = val_mae
        best_num_units = num_units

print("Best number of units:", best_num_units)


###TUNE THE LEARNING RATE
###TUNE THE LEARNING RATE
# Set the learning rate scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(int(epoch) / 20), verbose=1)
# Initialize the optimizer
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8)
optimizer=tf.keras.optimizers.SGD(learning_rate=1e-08,momentum=0.9)
# Set the training parameters
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
# Train the model
history = model.fit(dataset_train, epochs=25, callbacks=[lr_schedule])

# Define the learning rate array
lrs = 1e-8 * (10 ** (np.arange(25) / 20))

# Set the figure size
plt.figure(figsize=(10, 6))
# Set the grid
plt.grid(True)
# Plot the loss in log scale
plt.semilogx(lrs, history.history["loss"])
# Increase the tickmarks size
plt.tick_params('both', length=10, width=1, which='both')
# Set the plot boundaries
plt.axis([1e-8, 1e-1, 0, 1])
plt.show()
'''
#########TRAIN THE MODEL WITH THE PROPER LEARNING RATE
# Reset states generated by Keras
#
#tf.keras.backend.clear_session()


# Compile the model
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=3e-08,momentum=0.9),metrics=["mae"])

# Train the model with the combined dataset
# ... (previous code)

# Train the model with the combined dataset
history = model.fit(dataset_train, epochs=25, validation_data=dataset_valid)

# Save the model weights after training
model.save_weights('model_weights.h5')

# Later, when you want to use the trained model

#new_model.load_weights('model_weights1.h5')

# Plot training and validation loss over epochs
plt.figure(figsize=(12, 4))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation MAE over epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE', color='blue')
plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
plt.title('Training and Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# ... (rest of the code)
# Evaluate the model on the validation dataset
evaluation_result = model.evaluate(dataset_valid)
print("Validation Loss:", evaluation_result)


# Predict on the validation dataset
# ... (previous code)

# Predict on the validation dataset
# Predict on the validation dataset
# Predict on the validation dataset
num_predictions_beyond_validation = 2
validation_predictions = []

# Use the last window_size elements from the training set to seed the predictions
current_window = x_train_scaled[-window_size:]

# Adjust time steps for validation predictions
validation_time_steps = np.arange(len(x_valid_scaled))

for time in range(len(validation_time_steps) + num_predictions_beyond_validation):
    # ... (rest of your prediction loop)

    # Reshape the current_window to match the model's input shape
    current_window_reshaped = np.array(current_window[-window_size:]).reshape(1, -1)

    # Predict the next value using the model on the validation dataset
    predicted_value_scaled = model.predict(current_window_reshaped)[0, 0]

    # Append the predicted value to the list of predictions
    validation_predictions.append(predicted_value_scaled)

    # Print the actual and predicted values during validation along with the time step
    if time < len(x_valid_scaled):
        actual_value_scaled = x_valid_scaled[time]
        # Denormalize the actual value
        actual_value_denormalized = scaler.inverse_transform(np.array(actual_value_scaled).reshape(1, -1)).flatten()
        # Denormalize the predicted value
        predicted_value_denormalized = scaler.inverse_transform(np.array(predicted_value_scaled).reshape(1, -1)).flatten()
        print(f'Time: {time_valid[time]}, Actual: {actual_value_denormalized}, Predicted: {predicted_value_denormalized}')

    # Update the current window for the next iteration using the true value from the validation set
    if time < len(x_valid_scaled):
        current_window = np.append(current_window, x_valid_scaled[time])[1:]
    else:
        # Print the predicted value beyond validation along with the time step
        # Denormalize the predicted value beyond validation
        predicted_value_denormalized = scaler.inverse_transform(np.array(predicted_value_scaled).reshape(1, -1)).flatten()
        print(f'Time: {time_valid[-1] + time - len(x_valid_scaled) + 1}, Predicted (Beyond Validation): {predicted_value_denormalized}')
        current_window = np.append(current_window, predicted_value_scaled)[1:]


# Plot the original data, training data, validation data, and predictions
plt.figure(figsize=(10, 6))

# Plot original data in blue
plt.plot(time_step, dummy_data, label='Original Data', marker='o', linestyle='-', color='black')

# Plot training data in green
plt.plot(time_step[:split_time], dummy_data[:split_time], label='Training Data', color='blue')

# Plot validation data in orange
plt.plot(time_valid, dummy_data[split_time:], label='Validation Data', color='red')

# Denormalize validation predictions
validation_predictions_denormalized = scaler.inverse_transform(np.array(validation_predictions).reshape(-1, 1)).flatten()

# Plot predictions (Validation)
plt.plot(time_valid, validation_predictions_denormalized[:len(time_valid)], label='Predictions (Validation)', color='orange')

# Highlight the last few predictions beyond the validation set in red
last_predictions_beyond_validation_denormalized = scaler.inverse_transform(np.array(validation_predictions[-num_predictions_beyond_validation:]).reshape(-1, 1)).flatten()
time_last_predictions_beyond_validation = np.arange(split_time + len(time_valid), split_time + len(time_valid) + num_predictions_beyond_validation)
plt.scatter(time_last_predictions_beyond_validation, last_predictions_beyond_validation_denormalized, color='red', marker='X', label='Last Predictions Beyond Validation')

plt.legend()
plt.title('Original Data, Training Data, Validation Data, and Predictions')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.show()

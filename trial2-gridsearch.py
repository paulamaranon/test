import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# Assuming you have the following variables defined
claims = np.array([7300, 7700, 8000, 5700, 7800, 7600, 8000, 6700, 6200, 6800, 6700, 5800, 6200, 5500])
time_step = np.arange(len(claims))
split_time = 10

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
    features = []
    labels = []
    for i in range(len(series) - window_size):
        window = series[i: i + window_size]
        target = series[i + window_size]
        features.append(window)
        labels.append(target)
    return np.array(features), np.array(labels)


# Set window size
window_size = 3  # Choose a larger window size

# Create windowed dataset for training
X_train_np, y_train_np = windowed_dataset(x_train_scaled, window_size)

# Create windowed dataset for validation
X_valid_np, y_valid_np = windowed_dataset(x_valid_scaled, window_size)

# Use TimeSeriesSplit for time series data
tscv = TimeSeriesSplit(n_splits=5)


# Define a function to create and train the model with given hyperparameters
def create_and_train_model(rnn_units):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[window_size]),
        tf.keras.layers.SimpleRNN(rnn_units),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')

    # Train the model
    model.fit(X_train_np, y_train_np, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model
    score = model.evaluate(X_valid_np, y_valid_np, verbose=0)
    return -score  # Negative because GridSearchCV looks for the maximum score


# Perform manual grid search
param_grid = {'rnn_units': [32, 64, 128, 256, 320]}
best_score = float('-inf')
best_params = None

for rnn_units in param_grid['rnn_units']:
    score = create_and_train_model(rnn_units)
    print(f"RNN Units: {rnn_units}, Validation Score: {score}")

    if score > best_score:
        best_score = score
        best_params = {'rnn_units': rnn_units}

# Print the best parameters and corresponding score
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)

''' 
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[window_size]),
    tf.keras.layers.SimpleRNN(72),  # Remove return_sequences=True
    tf.keras.layers.Dense(1)
])

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(72, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.Dropout(0.1),  # Add dropout layer
    tf.keras.layers.Dense(1)
])


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                      strides=1,
                      activation="relu",
                      padding='causal',
                      input_shape=[window_size, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1)
  #tf.keras.layers.Lambda(lambda x: x * 100)
])

# Build the Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                     input_shape=[window_size]),
  tf.keras.layers.SimpleRNN(40, return_sequences=True),
  tf.keras.layers.SimpleRNN(40),
  tf.keras.layers.Dense(1)
  #tf.keras.layers.Lambda(lambda x: x * 100.0)
])


# Compile the model
#model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
# Set the learning rate
learning_rate = 1e-5
# Set the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
#optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer)

#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the model
history = model.fit(dataset_train, epochs=40, validation_data=dataset_valid)#, callbacks=[early_stopping])


# Plot the original data, training data, test data, and predictions
plt.figure(figsize=(12, 6))

# Plot original data
plt.subplot(2, 1, 1)
plt.plot(time_step, claims, label='Original Data', marker='o', linestyle='-', color='black')

# Plot training data (inverse transform x_train_scaled)
plt.plot(time_step[:split_time], scaler.inverse_transform(x_train_scaled.reshape(-1, 1)).flatten(), label='Training Data', color='blue')

# Plot validation data (inverse transform x_valid_scaled)
plt.plot(time_step[split_time:], scaler.inverse_transform(x_valid_scaled.reshape(-1, 1)).flatten(), label='Validation Data', color='red')

# Plot predictions on validation set for the entire duration
forecast_valid_scaled = model.predict(dataset_valid)

# Flatten the predictions and inverse transform
forecast_valid = scaler.inverse_transform(forecast_valid_scaled.flatten().reshape(-1, 1)).flatten()

# Use the same indexing as for validation data
time_forecast_valid = time_step[split_time:split_time + len(forecast_valid)]

# Ensure dimensions match
assert len(time_forecast_valid) == len(forecast_valid), "Dimensions do not match"

# Plot predictions on validation set along the same time steps
plt.plot(time_forecast_valid, forecast_valid, label='Predictions', linestyle='--', color='green')

# Extend predictions one month ahead
for i in range(3):  # Adjust the number time steps to the future
    # Predict one time step ahead
    forecast_future_scaled = model.predict(np.array([x_valid_scaled[-window_size:]]))
    forecast_future = scaler.inverse_transform(forecast_future_scaled.flatten().reshape(-1, 1)).flatten()

    # Update the time and forecast arrays
    time_forecast_valid = np.append(time_forecast_valid, time_forecast_valid[-1] + 1)
    forecast_valid = np.append(forecast_valid, forecast_future)

    # Plot the updated predictions
    plt.plot(time_forecast_valid, forecast_valid, linestyle='--', color='green', alpha=0.5)  # Adjust alpha for transparency

    # Update the input sequence for the next prediction
    x_valid_scaled = np.append(x_valid_scaled, forecast_future[-1])


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
'''
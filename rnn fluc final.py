import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from scipy.stats import gaussian_kde

# Load data
data = pd.read_csv('spf500.csv', delimiter=',', header=None, names=['date', 'price'])
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna()

# Select data within 2008-01-01 to 2018-12-31
data = data[(data['date'] >= '2008-01-01') & (data['date'] <= '2018-12-31')].reset_index(drop=True)

# Apply differencing to stabilize the time series
data['price_diff'] = data['price'].diff().fillna(0)

# Normalize differenced price data
scaler = MinMaxScaler(feature_range=(0, 1))
data['scaled_diff'] = scaler.fit_transform(data[['price_diff']])

# Split data
split_2008_2016 = data[data['date'] <= '2016-12-31']
split_2017_2018 = data[data['date'] > '2016-12-31']

# Train (80%) and Validation (20%) from 2008-2016
split_idx = int(len(split_2008_2016) * 0.8)
train_data = split_2008_2016.iloc[:split_idx]
val_data = split_2008_2016.iloc[split_idx:]
test_data = split_2017_2018

seq_length = 7
predict_length=5
# Prepare sequences

np.random.seed(28)
tf.random.set_seed(28)
random.seed(28)
def create_sequences(data, seq_length, predict_length):
	X, y = [], []
	for i in range(len(data) - seq_length - predict_length + 1):
		X.append(data[i:i + seq_length])
		y.append(data[i + seq_length:i + seq_length + predict_length])
	return np.array(X), np.array(y)


train_scaled_diff = train_data['scaled_diff'].values
val_scaled_diff = val_data['scaled_diff'].values
test_scaled_diff = test_data['scaled_diff'].values

X_train, y_train = create_sequences(train_scaled_diff, seq_length,predict_length)
X_val, y_val = create_sequences(val_scaled_diff, seq_length,predict_length)

X_train = X_train.reshape(X_train.shape[0], seq_length, 1)
X_val = X_val.reshape(X_val.shape[0], seq_length, 1)

# Build model
def build_rnn_model():
    model = Sequential([
        SimpleRNN(250, activation='tanh', input_shape=(seq_length, 1)),
        Dense(predict_length)  # Predicting single difference value
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_rnn_model()
model.fit(X_train, y_train, epochs=40, batch_size=24, validation_data=(X_val, y_val), verbose=1)

# Compute residuals
train_predictions = model.predict(X_train, verbose=0).flatten()
# Flatten y_train and train_predictions to make them compatible for residual calculation
y_train_flat = y_train.flatten()
train_predictions_flat = train_predictions.flatten()

# Now compute residuals
train_residuals = y_train_flat - train_predictions_flat
# Fit Gaussian KDE on residuals
kde = gaussian_kde(train_residuals)

# Predict on test data with fluctuation simulation
# Initialize list to store predictions
predictions = []

# Start with the first 7 values from the validation set
current_sequence = val_scaled_diff[:seq_length]

# Iterate over the validation set
for i in range(len(test_data)):
	# Predict the next 5 values based on the current 7 values
	predicted_diff = model.predict(current_sequence.reshape(1, seq_length, 1), verbose=0)[0]

	# Append the predicted 5 values
	predictions.append(predicted_diff)

	# Slide the window: remove the first value and add the predicted 5 values
	current_sequence = np.append(current_sequence[1:], predicted_diff)[:seq_length]

# Flatten the predictions to get one value per time step
predictions_flat = np.array(predictions).flatten()

# Simulate fluctuations (adjust for 5 predicted values)
simulated_fluctuations = kde.resample(len(predictions) * predict_length).flatten()  # Resample for 5 values per prediction
predictions_with_fluctuations = np.array(predictions) + simulated_fluctuations.reshape(len(predictions), predict_length)

# Denormalize the predictions (predicted differences)
predictions_diff = scaler.inverse_transform(predictions_with_fluctuations.flatten().reshape(-1, 1)).flatten()

# Get the last price from validation data (ensuring smooth continuation)
last_train_value = val_data['price'].iloc[-1]

# Invert differencing for 5 values prediction
predicted_prices = []
predicted_prices.append(last_train_value)

# Reconstruct the prices using the cumulative sum of the predictions
for i in range(0, len(predictions_diff), predict_length):
	predicted_prices.extend(np.cumsum(predictions_diff[i:i + predict_length]) + predicted_prices[-1])

# Ensure the predicted prices match the length of the test data
predicted_prices = predicted_prices[:len(test_data)]

# Create a pandas Series with the predicted prices and correct dates
predicted_prices = pd.Series(predicted_prices, index=test_data['date'])

# Evaluate performance (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['price'], predicted_prices))
mae = mean_absolute_error(test_data['price'], predicted_prices)
print(f'Test RMSE: {rmse}')
print(f'MAE: {mae}')

# Plot full dataset with predictions overlayed
plt.figure(figsize=(14, 7))

# Plot actual prices for the entire dataset
plt.plot(data['date'], data['price'], label='Actual Price (Full Data)', color='blue', linewidth=2)

# Overlay predicted prices for the test period
plt.plot(predicted_prices.index, predicted_prices, label='Predicted Price (Test Period)', color='red', linewidth=2)

# Highlight the test data for better visualization
plt.axvline(x=test_data['date'].iloc[0], color='black', linestyle='--', label='Test Start')
plt.axvline(x=test_data['date'].iloc[-1], color='black', linestyle='--')

# Labels and legend
plt.xlabel('Date',fontsize=22, fontweight='bold')
plt.ylabel('Stock Price', fontsize=22, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(fontsize=18, prop={'weight': 'bold'})
plt.show()


# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], test_data['price'], label='Actual Price', color='blue', linewidth=2)
plt.plot(predicted_prices.index, predicted_prices, label='Predicted Price', color='red', linewidth=2)
plt.legend()
plt.show()

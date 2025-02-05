import numpy as np
import pandas as pd
import random
import matplotlib
#matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load data
data = pd.read_csv('spf500.csv', delimiter=',', header=None, names=['date', 'price'])
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna()

# Select data within 2008-01-01 to 2018-12-31
data = data[(data['date'] >= '2008-01-01') & (data['date'] <= '2018-12-31')].reset_index(drop=True)

# Normalize price data
scaler = MinMaxScaler(feature_range=(0, 1))
data['scaled_price'] = scaler.fit_transform(data[['price']])

# Split data
split_2008_2016 = data[data['date'] <= '2016-12-31']
split_2017_2018 = data[data['date'] > '2016-12-31']

# Train (80%) and Validation (20%) from 2008-2016
split_idx = int(len(split_2008_2016) * 0.8)
train_data = split_2008_2016['scaled_price'][:split_idx].values
val_data = split_2008_2016['scaled_price'][split_idx:].values

# Test data (2017-2018)
test_data = split_2017_2018['scaled_price'].values


def create_sequences(data, seq_length, predict_length):
	X, y = [], []
	for i in range(len(data) - seq_length - predict_length + 1):
		X.append(data[i:i + seq_length])
		y.append(data[i + seq_length:i + seq_length + predict_length])
	return np.array(X), np.array(y)


# Parameters
seq_length = 7
predict_length = 5


def build_lstm_model():
	model = Sequential([
		SimpleRNN(250, activation='tanh', input_shape=(seq_length, 1)),
		Dense(predict_length)
	])
	# model = Sequential([
	# 	SimpleRNN(250, activation='tanh', return_sequences=True, input_shape=(seq_length, 1)),  # First RNN layer
	# 	SimpleRNN(250, activation='tanh', return_sequences=True),  # Second RNN layer
	# 	# Third RNN layer (last one, no return_sequences)
	# 	Dense(predict_length)  # Output layer
	# ])


	model.compile(optimizer='adam', loss='mean_squared_error')
	return model


# Create sequences
X_train, y_train = create_sequences(train_data, seq_length, predict_length)
X_val, y_val = create_sequences(val_data, seq_length, predict_length)

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# Store results
rmse_list = []
mae_list = []

'''start 20 plot'''
plt.figure(figsize=(12, 6))

# Plot actual test data
actuals = test_data.reshape(-1, 1)  # Convert test data to array
actuals = scaler.inverse_transform(actuals)  # Denormalize
plt.plot(split_2017_2018['date'][:len(actuals)], actuals, label='Actual Price', color='blue', linewidth=2)

all_predictions=[]
''''''
# Run model for 10 different seeds
random_seeds = random.sample(range(0, 300), 2)
for seed in random_seeds:
#
# #
# 	np.random.seed(seed)
# 	tf.random.set_seed(seed)
# 	random.seed(seed)

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    model = build_lstm_model()

	# Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

	# Train model
    history = model.fit(X_train, y_train, epochs=100, batch_size=24, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])
	#history = model.fit(X_train, y_train, epochs=40, batch_size=24, verbose=1, validation_data=(X_val, y_val))

	# Rolling prediction on test set
    predictions = []
    current_sequence = val_data[-seq_length:].tolist()

    while len(predictions) < len(test_data):
        predicted_values = model.predict(np.array(current_sequence).reshape(1, seq_length, 1), verbose=0)[0]
        predictions.extend(predicted_values)
        current_sequence.extend(predicted_values)
        current_sequence = current_sequence[predict_length:]

	# Trim predictions to match test_data length
    predictions = np.array(predictions[:len(test_data)]).reshape(-1, 1)
    actuals = test_data.reshape(-1, 1)

	# Denormalize
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)

	# Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
	#print(f'seed: {seed}')
    rmse_list.append(rmse)
    mae_list.append(mae)

	#'''start 20 plot'''
	# Store each run's predictions
    all_predictions.append(predictions)

	# Plot each run's predictions
    plt.plot(split_2017_2018['date'][:len(predictions)], predictions, color='red', alpha=0.3)  # Red, semi-transparent

# Compute average metrics
avg_rmse = np.mean(rmse_list)
avg_mae = np.mean(mae_list)
print(f'without fluctuations')
print(f'Average RMSE over 20 runs: {avg_rmse}')
print(f'Average MAE over 20 runs: {avg_mae}')

# Labels and title
plt.xlabel('Date',fontsize=23, fontweight='bold' )
plt.ylabel('Stock Price', fontsize=23, fontweight='bold')
# Select specific years for x-axis labels
xtick_labels = ['2017', '2018', '2019']
xtick_positions = [pd.Timestamp(year=int(year), month=1, day=1) for year in xtick_labels]

plt.xticks(xtick_positions, xtick_labels, fontsize=23, fontweight='bold')
plt.yticks(fontsize=23, fontweight='bold')
# plt.title('S&P 500 Test Data: Actual vs. 20 Prediction Runs')

# Add legend
plt.legend(['Actual Price', 'Predictions'],fontsize=23)

# Show plot
plt.show()
''''''



# Plot the full dataset
plt.figure(figsize=(12, 6))
plt.plot(data['date'], scaler.inverse_transform(data[['scaled_price']]), label='Actual Price', color='blue')

# Highlight training data
#plt.axvline(x=data.iloc[split_idx]['date'], color='green', linestyle='--', label='Validation Split')

# Highlight test data range
plt.axvline(x=split_2017_2018.iloc[0]['date'], color='black', linestyle='--', label='Test Start')

# Plot predictions on the test set
test_dates = split_2017_2018['date'].values[:len(predictions)]
plt.plot(test_dates, predictions, label='Predictions', color='red', linestyle='--')

# Labels and legend
plt.xlabel('Date',fontsize=23, fontweight='bold')
plt.ylabel('Stock Price', fontsize=23, fontweight='bold')
# Select specific years for x-axis labels
xtick_labels = ['2017', '2018', '2019']
xtick_positions = [pd.Timestamp(year=int(year), month=1, day=1) for year in xtick_labels]

plt.xticks(xtick_positions, xtick_labels, fontsize=23, fontweight='bold')

plt.yticks(fontsize=23, fontweight='bold')
plt.legend(fontsize=20)
plt.show()
#prop={'weight': 'bold'}

# plt.figure(figsize=(10, 5))
#
# # Plot actual test data
# plt.plot(split_2017_2018['date'][:len(actuals)], actuals, label='Actual Price', color='blue')
#
# # Plot predictions
# plt.plot(split_2017_2018['date'][:len(predictions)], predictions, label='Predicted Price', color='red', linestyle='dashed')
#
# # Labels and title
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.title('S&P 500 Test Data: Actual vs Predictions')
# plt.legend()
# plt.show()

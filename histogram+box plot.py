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
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# Run model for 100 different seeds
for _ in range(2):
    model = build_lstm_model()
    
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train model
    history = model.fit(X_train, y_train, epochs=100, batch_size=24, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
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
    
    rmse_list.append(rmse)
    mae_list.append(mae)
    
# Compute statistics
rmse_max = np.max(rmse_list)
rmse_min = np.min(rmse_list)
mae_max = np.max(mae_list)
mae_min = np.min(mae_list)

### **RMSE 直方图 + Box Plot** ###
fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})  # 1:1 让两者一样大

# 直方图
ax[0].hist(rmse_list, bins=40, density=True, alpha=0.6, color='blue', label='RMSE Histogram')
mean_rmse = np.mean(rmse_list)
std_rmse = np.std(rmse_list)
x = np.linspace(min(rmse_list), max(rmse_list), 300)
#pdf = norm.pdf(x, mean_rmse, std_rmse)
#ax[0].plot(x, pdf, color='red', linewidth=2, label='Gaussian Curve')
ax[0].set_xlabel('RMSE', fontsize=20, fontweight='bold')
ax[0].set_ylabel('Probability Density', fontsize=20, fontweight='bold')
ax[0].legend(fontsize=12, prop={'weight': 'bold'})  # Set legend font size and bold

y_min, y_max = ax[0].get_ylim()
ax[0].set_yticks([y_min, (y_min + y_max) / 2, y_max])  # Set three evenly spaced tick values

# Make ticks bold
ax[0].tick_params(axis='both', which='both', labelsize=18, width=2)  # Set tick label size and width
for label in ax[0].get_xticklabels() + ax[0].get_yticklabels():
    label.set_fontweight('bold')  # Set bold font weight


#ax[0].grid()

# Box plot 
ax[1].boxplot(rmse_list, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
ax[1].set_xlabel('RMSE',fontsize=20, fontweight='bold')
ax[1].set_yticks([])
# Make x-axis tick labels bold
for label in ax[1].get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(18)
#ax[1].grid()

#plt.suptitle('RMSE Histogram and Box Plot (Equal Size)')
plt.tight_layout()
plt.show()

### **MAE 直方图 + Box Plot** ###
fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})  # 让两者一样大

# 直方图
ax[0].hist(mae_list, bins=40, density=True, alpha=0.6, color='green', label='MAE Histogram')
mean_mae = np.mean(mae_list)
std_mae = np.std(mae_list)
x = np.linspace(min(mae_list), max(mae_list), 300)
#pdf = norm.pdf(x, mean_mae, std_mae)
#ax[0].plot(x, pdf, color='red', linewidth=2, label='Gaussian Curve')
ax[0].set_xlabel('MAE', fontsize=20, fontweight='bold')
ax[0].set_ylabel('Probability Density', fontsize=20, fontweight='bold')
ax[0].legend(fontsize=12, prop={'weight': 'bold'})  # Set legend font size and bold

y_min, y_max = ax[0].get_ylim()
ax[0].set_yticks([y_min, (y_min + y_max) / 2, y_max])  # Set three evenly spaced tick values

# Make ticks bold
ax[0].tick_params(axis='both', which='both', labelsize=18, width=2)  # Set tick label size and width
for label in ax[0].get_xticklabels() + ax[0].get_yticklabels():
    label.set_fontweight('bold')  # Set bold font weight

#ax[0].grid()

# Box plot 
ax[1].boxplot(mae_list, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
ax[1].set_xlabel('MAE',fontsize=20, fontweight='bold')
ax[1].set_yticks([])
# Make x-axis tick labels bold
for label in ax[1].get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(18)
#ax[1].grid()

#plt.suptitle('MAE Histogram and Box Plot (Equal Size)')
plt.tight_layout()
plt.show()

print(f'Average RMSE over 100 runs: {np.mean(rmse_list)}')
print(f'Average MAE over 100 runs: {np.mean(mae_list)}')
print(f'Max RMSE: {rmse_max}, Min RMSE: {rmse_min}')
print(f'Max MAE: {mae_max}, Min MAE: {mae_min}')


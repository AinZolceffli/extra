#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:38:58 2025

@author: lixinran
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Load data
data = pd.read_csv('sp500_daily_prices.csv', delimiter=',', header=None, names=['date', 'price'])
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
seq_length = 14
predict_length = 5

def build_lstm_model():
    model = Sequential([
        LSTM(250, activation='tanh', input_shape=(seq_length, 1)),
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
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


### **Compute RMSE and MAE Statistics**
mean_rmse = round(np.mean(rmse_list), 2)
median_rmse = round(np.median(rmse_list), 2)
iqr_rmse = round(iqr(rmse_list), 2)

mean_mae = round(np.mean(mae_list), 2)
median_mae = round(np.median(mae_list), 2)
iqr_mae = round(iqr(mae_list), 2)

### **RMSE Histogram + Box Plot**
fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

# Histogram (light blue)
ax[0].hist(rmse_list, bins=20, density=True, alpha=0.6, 
           color='#87CEFA', edgecolor='black', linewidth=1.5, label='RMSE Histogram')

ax[0].set_xlabel('RMSE', fontsize=23, fontweight='bold')
ax[0].set_ylabel('Probability Density', fontsize=20, fontweight='bold')
ax[0].legend(fontsize=23)

# Box plot (light blue)
ax[1].boxplot(rmse_list, vert=False, patch_artist=True, boxprops=dict(facecolor='#87CEFA'))
ax[1].set_xlabel('RMSE', fontsize=23, fontweight='bold')
ax[1].set_yticks([])

plt.tight_layout()  # Fix layout issues
plt.show()


### **MAE Histogram + Box Plot**
fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

# Histogram (light green)
ax[0].hist(mae_list, bins=20, density=True, alpha=0.6, 
           color='#90EE90', edgecolor='black', linewidth=1.5, label='MAE Histogram')

ax[0].set_xlabel('MAE', fontsize=23, fontweight='bold')
ax[0].set_ylabel('Probability Density', fontsize=20, fontweight='bold')
ax[0].legend(fontsize=23)

# Box plot (light green)
ax[1].boxplot(mae_list, vert=False, patch_artist=True, boxprops=dict(facecolor='#90EE90'))
ax[1].set_xlabel('MAE', fontsize=23, fontweight='bold')
ax[1].set_yticks([])

plt.tight_layout()  # Fix layout issues
plt.show()

### **Print Summary Statistics**
print(f'Average RMSE: {mean_rmse}')
print(f'Median RMSE: {median_rmse}')
print(f'IQR RMSE: {iqr_rmse}')
#print(f'Max RMSE: {round(rmse_max, 2)}, Min RMSE: {round(rmse_min, 2)}\n')

print(f'Average MAE: {mean_mae}')
print(f'Median MAE: {median_mae}')
print(f'IQR MAE: {iqr_mae}')
#print(f'Max MAE: {round(mae_max, 2)}, Min MAE: {round(mae_min, 2)}')

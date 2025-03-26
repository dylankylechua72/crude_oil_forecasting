import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('c:/wamp64/www/Personal_projects/crude_forecasting/crude_oil_prices.csv', parse_dates=['Date'], index_col='Date')
data['Price'] = data['Price'].ffill()

# Scale the data to [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Price']].values)

# Create sequences for LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])
        y.append(data[i+window_size, 0])
    return np.array(X), np.array(y)

window_size = 10  # Use past 10 days to predict the next day
X, y = create_sequences(scaled_data, window_size)

# Reshape X to be [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into training and testing sets (80/20 split)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

# Make predictions on test set
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Forecast future values
future_days = 30  # Predict the next 30 days
future_predictions = []
last_sequence = X_test[-1]  # Start with the last observed sequence

for _ in range(future_days):
    next_pred = model.predict(last_sequence.reshape(1, window_size, 1))
    future_predictions.append(next_pred[0, 0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_pred

# Inverse transform future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices', color='orange')
plt.axvline(len(y_test_inv), color='red', linestyle='dashed', label='Forecast Start')
plt.plot(range(len(y_test_inv), len(y_test_inv) + future_days), future_predictions, label='Future Predictions', color='green')
plt.title('LSTM Forecast of Crude Oil Price')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()

# Print RMSE
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
print("LSTM Forecast RMSE:", rmse)
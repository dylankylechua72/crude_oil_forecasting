import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('c:/wamp64/www/Personal_projects/crude_forecasting/crude_oil_prices.csv', parse_dates=['Date'], index_col='Date')
data = data.asfreq('D')  # Resample to daily frequency
data['Price'] = data['Price'].ffill()  # Forward fill missing values


# Create lag features
for lag in range(1, 6):
    data[f'lag_{lag}'] = data['Price'].shift(lag)

data.dropna(inplace=True)

# Prepare training data
X = data[[f'lag_{lag}' for lag in range(1, 6)]]
y = data['Price']

# Split into training and test sets
train_size = int(0.8 * len(data))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train the XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("XGBoost Forecast RMSE:", rmse)

# Generate future forecasts for 30 days
future_days = 30
last_known_values = X.iloc[-1].values  # Start from last known data
future_predictions = []

if pd.isna(data.index[-1]):
    raise ValueError("Last date in dataset is missing (NaT). Check the CSV file.")

last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)


for date in future_dates:
    next_pred = model.predict(last_known_values.reshape(1, -1))[0]
    future_predictions.append(next_pred)

    # Update the last_known_values with the new prediction
    last_known_values = np.roll(last_known_values, -1)  # Shift values
    last_known_values[-1] = next_pred  # Add new prediction

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, predictions, label='Predicted', color='green')
plt.plot(future_dates, future_predictions, label='Future Forecast', color='red', linestyle='dashed')
plt.title('XGBoost Forecast of Crude Oil Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

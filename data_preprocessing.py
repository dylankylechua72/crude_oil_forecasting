# data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset (make sure the CSV file is in the working directory)
data = pd.read_csv('c:/wamp64/www/Personal_projects/crude_forecasting/crude_oil_prices.csv')


# Convert 'Date' column to datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# Optional: Fill missing values (forward-fill)
data['Price'] = data['Price'].ffill()

# Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(data.index, data['Price'], label='Crude Oil Price')
plt.title('Crude Oil Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

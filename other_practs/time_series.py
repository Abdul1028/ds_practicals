import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Generate sample time series data
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
data = np.random.randn(len(dates))
ts = pd.Series(data, index=dates)

# Plot the time series data
ts.plot(figsize=(10, 5))
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Sample Time Series Data')
plt.show()

# Check stationarity using Augmented Dickey-Fuller test
result = adfuller(ts)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {:.3f}'.format(key, value))

# Plot ACF and PACF
plot_acf(ts, lags=30)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(ts, lags=30)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Fit ARIMA model
model = ARIMA(ts, order=(1, 1, 1))  # Example: ARIMA(1,1,1) model
results = model.fit()

# Forecast
forecast = results.forecast(steps=30)

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(ts, label='Observed')
plt.plot(pd.date_range(start=ts.index[-1], periods=30, freq='D'), forecast, label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Forecast')
plt.legend()
plt.show()

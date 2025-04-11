# "On the Introduction of Time Series Forecasting: A Case Study of Elspot Electricity Prices"

# Author: Alex Elkjær Vasegaard, Ph.D.
# Institute: Aalborg University
# Date: 2025-04-08

# Install necessary libraries before running:
# pip install nordpool pandas matplotlib statsmodels scikit-learn scipy tqdm

from nordpool import elspot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import power_transform
from scipy.stats import jarque_bera
import tqdm
import time
import warnings
warnings.filterwarnings("ignore")



# STEP 1: Load Elspot data (DK1) using nordpool library
def fetch_hourly_elspot_prices(area='DK1', start_date='2025-03-01', end_date='2025-04-01'):
    prices_spot = elspot.Prices()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    all_data = []

    current = end
    while current >= start:
        print(f"Fetching data for: {current.strftime('%Y-%m-%d')}")
        try:
            data = prices_spot.hourly(end_date=current.strftime('%Y-%m-%d'), areas=[area])
            for entry in data['areas'][area]['values']:
                if entry['value'] is not None:
                    all_data.append({
                        'time': pd.to_datetime(entry['start']),
                        'price': entry['value']
                    })
        except Exception as e:
            print(f"Failed to fetch data for {current.date()}: {e}")

        current -= pd.Timedelta(days=1)  # step back ~3 days (depends on API window)
        time.sleep(0.5)  # be polite to the API

    df = pd.DataFrame(all_data).drop_duplicates().sort_values('time').set_index('time')
    return df

# Use case
market = 'DK1'
elspot_hourly_df = fetch_hourly_elspot_prices(market, start_date='2025-03-01', end_date='2025-04-08')


# Q: How does my data look like?
elspot_hourly_df['price'].plot(figsize=(14, 4), title='Hourly Elspot Prices DK1 (EUR)')
plt.grid(True)
plt.show()









# STEP 2: EDA
# Q: Is something wrong with the data?
# Check for missing values or obvious issues
print("Missing values:", elspot_hourly_df['price'].isna().sum())

# Variance and mean
print("variance:", np.var(elspot_hourly_df['price']))
print("mean:", np.mean(elspot_hourly_df['price']))

# Q: Any outliers?
# Decompose using STL
stl = STL(elspot_hourly_df['price'])
res = stl.fit()

# Use the residuals to find outliers
residuals = res.resid
threshold = 3 # 3 standard deviations from mean
mask = np.abs(residuals) > threshold * residuals.std()
print('number of outliers:', sum(mask))

# Replace outliers with NaN or interpolate
elspot_hourly_df['cleanPrice'] = elspot_hourly_df['price']
elspot_hourly_df['cleanPrice'][mask] = np.nan
elspot_hourly_df['cleanPrice'] = elspot_hourly_df['cleanPrice'].interpolate(method='time')


# Q: How does my cleaned data look like?
elspot_hourly_df['cleanPrice'].plot(figsize=(14, 4), title='Cleaned and Transformed Hourly Elspot Prices DK1 (EUR)')
plt.grid(True)
plt.show()

# STL decomposition
stl = STL(elspot_hourly_df['cleanPrice'], seasonal=13)
res = stl.fit()
res.plot()
plt.suptitle("STL Decomposition of Elspot Prices")
plt.show()


# Histogram
plt.figure(figsize=(10, 5))
plt.hist(elspot_hourly_df['cleanPrice'], bins=50, edgecolor='black', alpha=0.7)
plt.title("Histogram of Cleaned Elspot Prices")
plt.xlabel("Price (EUR/MWh)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

#not normal?
jb_test = jarque_bera(elspot_hourly_df['cleanPrice'])
print("Jarque-Bera p-value (normality of residuals):", jb_test[1]) #-> not normal -> use models where this is not a requirement

# Difference data for further exploration
elspot_hourly_df['DiffPrice'] = elspot_hourly_df['cleanPrice'].diff().dropna()

# Q: How does my differenced data look like?
elspot_hourly_df['DiffPrice'].plot(figsize=(14, 4), title='Differenced Hourly Elspot Prices DK1 (EUR)')
plt.grid(True)
plt.show()

# ACF of differenced data
plot_acf(elspot_hourly_df['DiffPrice'].dropna())
plt.title("ACF of differenced Raw Prices")
plt.show()

# Clearly autocorrelation in the data! 







# STEP 3: Modelling
# Q: Do model assumptions fit?
adf_p = adfuller(elspot_hourly_df['DiffPrice'].dropna())[1]
print("ADF p-value:", adf_p, "=> Stationary?" , adf_p < 0.05)

# ACF 
plot_acf(elspot_hourly_df['DiffPrice'].dropna(), lags=23)
plt.title("ACF of differenced cleaned Prices")
plt.show()

# PACF 
plot_pacf(elspot_hourly_df['DiffPrice'].dropna(), lags=23)
plt.title("PACF of differenced cleaned Prices")
plt.show()

# Significant lags for acf and pacf values - with respect to confidence band assuming white noise
acf_vals = acf(elspot_hourly_df['DiffPrice'].dropna(), nlags=23)
pacf_vals = pacf(elspot_hourly_df['DiffPrice'].dropna(), nlags=23)

conf_bound = 1.96/np.sqrt(len(elspot_hourly_df['DiffPrice'].dropna()))

sig_acf_lags = [lag for lag, val in enumerate(acf_vals[1:], 1) if abs(val) > conf_bound]
sig_pacf_lags = [lag for lag, val in enumerate(pacf_vals[1:], 1) if abs(val) > conf_bound]

print("Significant ACF lags:", sig_acf_lags)
print("Significant PACF lags:", sig_pacf_lags)

# Suggest SARIMA(p,d,q)(P,D,Q,s) models
# We'll try a few combinations of significant p and q lags
# (Limit top significant lags for model suggestion)
top_p_lags = sorted(sig_pacf_lags, key=lambda x: abs(pacf_vals[x]), reverse=True)
top_q_lags = sorted(sig_acf_lags, key=lambda x: abs(acf_vals[x]), reverse=True) 
seasonal_orders = [(0, 1, 1, 24), (1, 1, 0, 24)]



# Prepare grid of combinations
model_grid = [
    (p, d, q, P, D, Q, s)
    for p in top_p_lags if p<15 # Note I have added the if statements 
    for q in top_q_lags if q<15 # to not make the analysis too large
    for d in [0,1] 
    for (P, D, Q, s) in seasonal_orders
]

# Note, we will not do the full analysis here.. only 10 models are analysed
model_grid = model_grid[:10]

# Evaluate models with progress bar
results = []
for (p, d, q, P, D, Q, s) in tqdm.tqdm(model_grid, desc="Fitting SARIMA models"):
    try:
        model = SARIMAX(elspot_hourly_df['cleanPrice'], #Note, we are back at the cleaned elspot prices
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=50)
        results.append({
            'order': (p, d, q),
            'seasonal_order': (P, D, Q, s),
            'AIC': res.aic,
            'BIC': res.bic
        })
    except Exception as e:
        print(f"Model ({p},{d},{q})({P},{D},{Q},{s}) failed: {e}")

# Show top models
results_df = pd.DataFrame(results).sort_values(by=['AIC', 'BIC']).reset_index(drop=True)
results_df.head()

# Therefore, we choose:
print('SARIMA model',results_df['order'][0],results_df['seasonal_order'][0])

model = SARIMAX(elspot_hourly_df['cleanPrice'], 
                order=results_df['order'][0],
                seasonal_order=results_df['seasonal_order'][0],
                enforce_stationarity=False,
                enforce_invertibility=False)
res = model.fit(disp=False, maxiter=50)
print(res.summary())







# STEP 4: Error term diagnostics
residuals = res.resid[1:]

# 1. Stationarity test (ADF)
adf_stat, adf_pvalue, _, _, _, _ = adfuller(residuals.dropna())
print(f"ADF test p-value: {adf_pvalue:.4f} => {'Stationary' if adf_pvalue < 0.05 else 'Not stationary'}")

# 2. Normality test (Jarque-Bera)
jb_stat, jb_pvalue = jarque_bera(residuals.dropna())
print(f"Jarque-Bera p-value: {jb_pvalue:.4f} => {'Normal residuals' if jb_pvalue > 0.05 else 'Not normal'}")

# 3. White noise check (visual ACF + Ljung-Box test)
fig, ax = plt.subplots(2, 1, figsize=(10, 5))
ax[0].plot(residuals)
ax[0].set_title("Model Residuals")
sm.graphics.tsa.plot_acf(residuals.dropna(), lags=40, ax=ax[1])
plt.tight_layout()
plt.show()

# Ljung-Box test for autocorrelation
ljung_box = sm.stats.acorr_ljungbox(residuals.dropna(), lags=[1,2,3,4,5,6,7,8,9,10], return_df=True)
print("\nLjung-Box test (up to lag 10):")
print(ljung_box)

# Summary
summary = {
    "ADF p-value": adf_pvalue,
    "Stationary?": adf_pvalue < 0.05,
    "Jarque-Bera p-value": jb_pvalue,
    "Normal?": jb_pvalue > 0.05,
    "Ljung-Box p-value (lag 10)": ljung_box.iloc[-1]['lb_pvalue'],
    "White noise?": ljung_box.iloc[-1]['lb_pvalue'] > 0.05
}

summary



# STEP 5: Testing Framework

def expanding_window_cv(series, order, seasonal_order, initial_train_size=200, horizon=24, step=24):
    """Performs expanding window cross-validation for SARIMA models.

    Args:
        series (pd.Series): Time series data (hourly).
        order (tuple): SARIMA (p,d,q).
        seasonal_order (tuple): Seasonal SARIMA (P,D,Q,s).
        initial_train_size (int): Number of initial observations to train on.
        horizon (int): Forecast horizon in hours.
        step (int): Step size between each test split (default: forecast horizon).

    Returns:
        pd.DataFrame: A DataFrame of RMSE and MAE for each test split.
    """
    metrics = []
    n = len(series)
    t = initial_train_size

    while t + horizon <= n:
        print('At day:', t/24, 'out of:', np.floor(n/24))
        train = series[:t]
        test = series[t:t + horizon]

        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=50)

            forecast = res.forecast(steps=horizon)

            rmse = np.sqrt(mean_squared_error(test, forecast))
            mae = mean_absolute_error(test, forecast)

            metrics.append({
                'train_end': series.index[t - 1],
                'rmse': rmse,
                'mae': mae
            })

        except Exception as e:
            print(f"Failed at split ending {series.index[t - 1]}: {e}")

        t += step

    return pd.DataFrame(metrics)

# Example usage with the previously simulated 'series'
cv_results = expanding_window_cv(
    series=elspot_hourly_df['cleanPrice'],
    order=results_df['order'][0],
    seasonal_order=results_df['seasonal_order'][0],
    initial_train_size=24*31, #31 days of training data (the rest for testing)
    horizon=24,
    step=24
)

# Plot results
cv_results.set_index('train_end')[['rmse', 'mae']].plot(title='Expanding Window Forecast Errors (24h Horizon)', figsize=(10, 4))
plt.grid(True)
plt.ylabel("Error")
plt.tight_layout()
plt.show()

cv_results.describe()




# The Last Hidden Step: Use The Model 
model = SARIMAX(elspot_hourly_df['cleanPrice'], 
                order=results_df['order'][0], seasonal_order=results_df['seasonal_order'][0],
                enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)

# Forecast next 24 hours
forecast_result = res.get_forecast(steps=24)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Plot
plt.figure(figsize=(12, 5))
plt.plot(elspot_hourly_df['price'].index[-96:], elspot_hourly_df['price'][-96:], label='Observed (last 96h)', color='black')
forecast_index = pd.date_range(start=elspot_hourly_df['price'].index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')
plt.plot(forecast_index, forecast_mean, label='Forecast (next 24h)', linestyle='--', color='blue')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='blue', alpha=0.2, label='95% Prediction Interval')
plt.title("24-Hour Forecast with 95% Prediction Interval")
plt.xlabel("Time")
plt.ylabel("Price (EUR/MWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





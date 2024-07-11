import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import theilslopes, kendalltau
from scipy.stats import theilslopes, kendalltau
from statsmodels.api import OLS, add_constant

def plot_sens_slope(data, title, ylabel, filename):
    X = np.arange(len(data))
    y = data.values.flatten()
    
    slope, intercept, lower, upper = theilslopes(y, X, 0.95)
    trend = intercept + slope * X
    tau, p_value = kendalltau(X, y)

    plt.plot(data.index, data, label='Time Series', color='blue')
    plt.plot(data.index, trend, 'r--', label=f'Trend Line (Slope: {slope:.4f})')
    plt.title(f"{title}\nMann-Kendall Tau: {tau:.4f}, p-value: {p_value:.4f}")
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_linear_regression(data, title, ylabel, filename):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values.flatten()
    
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    slope = model.coef_[0]

    X_ols = add_constant(X)
    ols_model = OLS(y, X_ols).fit()
    p_value = ols_model.pvalues[1]  # p-value for the slope

    plt.plot(data.index, data, label='Time Series', color='blue')
    plt.plot(data.index, trend, 'r--', label=f'Trend Line (Slope: {slope:.4f})')
    plt.title(f"{title}\n p-value: {p_value:.4f}")
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_first_half_vs_second_half(data, title, xlabel, ylabel, filename):
    first_half = data[data.index.year <= 1987]
    second_half = data[data.index.year >= 1988]
    
    # Ensure both halves have the same length by truncating the second half if necessary
    if len(second_half) > len(first_half):
        second_half = second_half.iloc[:len(first_half)]
    elif len(first_half) > len(second_half):
        first_half = first_half.iloc[:len(second_half)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(first_half, second_half, color='blue', label='Data points')
    
    min_val = min(min(first_half), min(second_half))
    max_val = max(max(first_half), max(second_half))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='45-degree line')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.clf()

max_data = pd.read_csv("daily_average_tmax.csv")
min_data = pd.read_csv("daily_average_tmin.csv")

max_data['time'] = pd.to_datetime(max_data["time"])
min_data['time'] = pd.to_datetime(min_data["time"])

max_data.set_index('time', inplace=True)
min_data.set_index('time', inplace=True)

combined = pd.concat([max_data, min_data], axis=1)
mean_data = combined.mean(axis=1)


yearly_max = max_data.resample('Y').max()
yearly_max_mean = max_data.resample('Y').mean()
yearly_min = min_data.resample('Y').min()
yearly_min_mean = min_data.resample('Y').mean()
yearly_mean = mean_data.resample('Y').mean()
monthly_max = max_data.resample('M').max()
monthly_max_mean = max_data.resample('M').mean()
monthly_min = min_data.resample('M').min()
monthly_min_mean = min_data.resample('M').mean()
monthly_mean = mean_data.resample('M').mean()

# LR
# Annual
plot_linear_regression(yearly_max, "Annual Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/annual_tmax_trend.png")
plot_linear_regression(yearly_max_mean, "Annual Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/annual_tmax_mean_trend.png")
plot_linear_regression(yearly_min, "Annual Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/annual_tmin_trend.png")
plot_linear_regression(yearly_min_mean, "Annual Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/annual_tmin_mean_trend.png")
plot_linear_regression(yearly_mean, "Annual Mean from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/annual_tmean_mean_trend.png")

# Monthly
plot_linear_regression(monthly_max, "Monthly Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/monthly_tmax_max_trend.png")
plot_linear_regression(monthly_max_mean, "Monthly Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/monthly_tmax_mean_trend.png")
plot_linear_regression(monthly_min, "Monthly Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/monthly_tmin_min_trend.png")
plot_linear_regression(monthly_min_mean, "Monthly Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/monthly_tmin_mean_trend.png")
plot_linear_regression(monthly_mean, "Monthly Mean from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/monthly_tmean_mean_trend.png")

# Daily
plot_linear_regression(max_data, "Daily Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/daily_tmax_max_trend.png")
plot_linear_regression(min_data, "Daily Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/daily_tmin_min_trend.png")
plot_linear_regression(mean_data, "Daily Mean from 1951 to 2023 across India", 'Value', "./all_india_trends/lr/daily_tmean_mean_trend.png")

# Sen's Slope
# Annual trends
plot_sens_slope(yearly_max, "Annual Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/annual_tmax_trend.png")
plot_sens_slope(yearly_max_mean, "Annual Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/annual_tmax_mean_trend.png")
plot_sens_slope(yearly_min, "Annual Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/annual_tmin_min_trend.png")
plot_sens_slope(yearly_min_mean, "Annual Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/annual_tmin_mean_trend.png")
plot_sens_slope(yearly_mean, "Annual Mean from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/annual_tmean_mean_trend.png")

# Monthly trends
plot_sens_slope(monthly_max, "Monthly Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/monthly_tmax_max_trend.png")
plot_sens_slope(monthly_max_mean, "Monthly Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/monthly_tmax_mean_trend.png")
plot_sens_slope(monthly_min, "Monthly Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/monthly_tmin_min_trend.png")
plot_sens_slope(monthly_min_mean, "Monthly Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/monthly_tmin_mean_trend.png")
plot_sens_slope(monthly_mean, "Monthly Mean from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/monthly_tmean_mean_trend.png")

# Daily trends
plot_sens_slope(max_data, "Daily Maxima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/tmax_max_trend.png")
plot_sens_slope(min_data, "Daily Minima from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/tmin_min_trend.png")
plot_sens_slope(mean_data, "Daily Mean from 1951 to 2023 across India", 'Value', "./all_india_trends/ss/tmean_mean_trend.png")

# Innovative slope
# Annual trends
plot_first_half_vs_second_half(yearly_max['tmax'], 
                               "First Half vs Second Half of Annual Maxima", 
                               "First Half (1951-1986)", 
                               "Second Half (1987-2023)", 
                               "./all_india_trends/innovative_slope/annual_max_first_vs_second_half.png")

plot_first_half_vs_second_half(yearly_max_mean['tmax'], 
                               "First Half vs Second Half of Annual Maxima", 
                               "First Half (1951-1986)", 
                               "Second Half (1987-2023)", 
                               "./all_india_trends/innovative_slope/annual_max_mean_first_vs_second_half.png")

plot_first_half_vs_second_half(yearly_min['tmin'], 
                               "First Half vs Second Half of Annual Minima", 
                               "First Half (1951-1986)", 
                               "Second Half (1987-2023)", 
                               "./all_india_trends/innovative_slope/annual_min_first_vs_second_half.png")

plot_first_half_vs_second_half(yearly_min_mean['tmin'], 
                               "First Half vs Second Half of Annual Minima", 
                               "First Half (1951-1986)", 
                               "Second Half (1987-2023)", 
                               "./all_india_trends/innovative_slope/annual_min_mean_first_vs_second_half.png")

plot_first_half_vs_second_half(yearly_mean, 
                               "First Half vs Second Half of Annual Mean", 
                               "First Half (1951-1986)", 
                               "Second Half (1987-2023)", 
                               "./all_india_trends/innovative_slope/annual_mean_first_vs_second_half.png")

# Monthly trends
plot_first_half_vs_second_half(monthly_max['tmax'], 
                               "First Half vs Second Half of Monthly Maxima", 
                               "First Half", 
                               "Second Half", 
                               "./all_india_trends/innovative_slope/monthly_max_first_vs_second_half.png")

plot_first_half_vs_second_half(monthly_max_mean['tmax'], 
                               "First Half vs Second Half of Monthly Maxima", 
                               "First Half", 
                               "Second Half", 
                               "./all_india_trends/innovative_slope/monthly_max_mean_first_vs_second_half.png")

plot_first_half_vs_second_half(monthly_min['tmin'], 
                               "First Half vs Second Half of Monthly Minima", 
                               "First Half", 
                               "Second Half", 
                               "./all_india_trends/innovative_slope/monthly_min_first_vs_second_half.png")

plot_first_half_vs_second_half(monthly_min_mean['tmin'], 
                               "First Half vs Second Half of Monthly Minima", 
                               "First Half", 
                               "Second Half", 
                               "./all_india_trends/innovative_slope/monthly_min_mean_first_vs_second_half.png")

plot_first_half_vs_second_half(monthly_mean, 
                               "First Half vs Second Half of Monthly Mean", 
                               "First Half", 
                               "Second Half", 
                               "./all_india_trends/innovative_slope/monthly_mean_first_vs_second_half.png")

# Daily trends
plot_first_half_vs_second_half(max_data['tmax'], 
                               "First Half vs Second Half of Daily Maxima", 
                               "First Half", 
                               "Second Half", 
                               "./all_india_trends/innovative_slope/daily_max_first_vs_second_half.png")

plot_first_half_vs_second_half(min_data['tmin'], 
                               "First Half vs Second Half of Daily Minima", 
                               "First Half", 
                               "Second Half", 
                               "./all_india_trends/innovative_slope/daily_min_first_vs_second_half.png")

plot_first_half_vs_second_half(mean_data, 
                               "First Half vs Second Half of Daily Mean", 
                               "First Half", 
                               "Second Half", 
                               "./all_india_trends/innovative_slope/daily_mean_first_vs_second_half.png")

import pandas as pd
import scipy.stats
import argparse
import matplotlib.pyplot as plt
import numpy as np
import imdlib as imd
import os
from sklearn.linear_model import LinearRegression
import matplotlib

matplotlib.use('Agg')

def get_grid_data(args, latitude, longitude):
    fulldata = pd.read_csv("./max_all.csv")
    griddata = fulldata[fulldata['latitude'] == latitude and fulldata['longitude'] == longitude]
    griddata.drop(columns = ["latitude", "longitude"], inplace = True)
    griddata['time'] = pd.to_datetime(griddata['time'])
    griddata.set_index('time', index = True)
    data = griddata[griddata.columns[0]]
    return data


def create_directories(name, latitude, longitude):
    paths = [
        f"./plots/{name}_{latitude:.2f}_{longitude:.2f}",
        f"./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}"
    ]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    return paths

def calculate_yearly_statistics(data):
    yearly_max = data.resample('Y').max()
    yearly_min = data.resample('Y').min()
    yearly_mean = data.resample('Y').mean()
    return yearly_max, yearly_min, yearly_mean

def plot_trend(yearly_data, title, y_label, output_path):
    X = np.arange(len(yearly_data)).reshape(-1, 1)
    y = yearly_data.values
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    plt.plot(yearly_data.index, yearly_data, label='Time Series', color='blue')
    plt.plot(yearly_data.index, trend, 'r--', label='Trend Line')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(output_path)
    plt.clf()

def save_fit_parameters(name, latitude, longitude, params, ll, dist_name):
    np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/params_mle_{dist_name}.npy', params)
    np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/ll_mle_{dist_name}.npy', ll)

def fit_distribution(yearly_max, dist, dist_name):
    params = dist.fit(yearly_max)
    ll = dist.logpdf(yearly_max, *params)
    return params, ll

def plot_distribution_comparison(yearly_max, params_mle_gev, params_mle_gumbel, output_path):
    x = np.linspace(min(yearly_max), max(yearly_max), 100)
    plt.plot(x, scipy.stats.gumbel_r.cdf(x, *params_mle_gumbel), 'r-', label='Gumbel MLE Fit')
    plt.plot(x, scipy.stats.genextreme.cdf(x, *params_mle_gev), 'b-', label='GEV MLE Fit')
    plt.title('Comparison of Gumbel and GEV Fits to EDF')
    plt.xlabel('Data values')
    plt.ylabel('Cumulative probability')
    plt.legend()
    plt.savefig(output_path)
    plt.clf()

def plot_full_data(data, output_path):
    data.plot()
    plt.title("Maximum Daily Temperatures from 1951 to 2023")
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.savefig(output_path)
    plt.clf()

def plot_histogram(data, title, output_path):
    fig, ax = plt.subplots()
    data.plot(kind='hist', ax=ax)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(output_path)
    plt.clf()

def plot_edf(data, output_path):
    data_sorted = data.sort_values(ascending=True)
    edf = np.arange(1, len(data_sorted)+1) / len(data_sorted)
    plt.figure(figsize=(8, 4))
    plt.step(data_sorted, edf, where="post", label='EDF')
    plt.xlabel('Value')
    plt.ylabel('EDF')
    plt.title('Empirical Distribution Function (EDF)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.clf()

def fit_and_plot_half_data(data, name, latitude, longitude):
    before_1989 = data[data.index.year <= 1989]
    after_1989 = data[data.index.year > 1989]

    plt.figure(figsize=(10, 6))

    fit_and_plot(before_1989, 'Before 1989', 'red', name, latitude, longitude)
    fit_and_plot(after_1989, 'After 1989', 'blue', name, latitude, longitude)

    plt.title('Comparison of Distribution Fits to Data Before and After 1989')
    plt.xlabel('Data values')
    plt.ylabel('Cumulative probability')
    plt.legend()
    plt.savefig(f'./plots/{name}_{latitude:.2f}_{longitude:.2f}/edf_fit_comparison_before_after_1989.png')
    plt.clf()

def fit_and_plot(data, label, color, name, latitude, longitude):
    params_mle = scipy.stats.genextreme.fit(data)
    params_gumbel_mle = scipy.stats.gumbel_r.fit(data)
    ll_mle = scipy.stats.genextreme.logpdf(data, *params_mle)
    ll_gumbel_mle = scipy.stats.genextreme.logpdf(data, *params_gumbel_mle)

    try:
        params_gumbel_mm = scipy.stats.gumbel_r.fit(data, method="MM")
        ll_gumbel_mm = scipy.stats.gumbel_r.logpdf(data, *params_gumbel_mm)
        np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/params_mm_gumbel_{label}.npy', params_gumbel_mm)
        np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/ll_mm_gumbel_{label}.npy', ll_gumbel_mm)
    except Exception as e:
        print(f"An error occurred during MM Gumbel fitting: {e}")
        params_gumbel_mm = None

    try:
        params_mm = scipy.stats.genextreme.fit(data, method="MM")
        ll_mm = scipy.stats.genextreme.logpdf(data, *params_mm)
        np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/params_mm_gev_{label}.npy', params_mm)
        np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/ll_mm_gev_{label}.npy', ll_mm)
    except Exception as e:
        print(f"An error occurred during MM GEV fitting: {e}")
        params_mm = None

    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, scipy.stats.genextreme.cdf(x, *params_mle), color=color, linestyle='-', label=f'GEV MLE Fit {label}')
    plt.plot(x, scipy.stats.gumbel_r.cdf(x, *params_gumbel_mle), color=color, linestyle='--', label=f'Gumbel MLE Fit {label}')

    np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/params_mle_gev_{label}.npy', params_mle)
    np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/params_mle_gumbel_{label}.npy', params_gumbel_mle)
    np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/ll_mle_gev_{label}.npy', ll_mle)
    np.save(f'./fit_parameters/{name}_{latitude:.2f}_{longitude:.2f}/ll_mle_gumbel_{label}.npy', ll_gumbel_mle)

def main():
    
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument('--name', type=str, required=True, help="City Name if applicable")
    parser.add_argument('--latitude', type=str, required=True, help="Grid Latitude Value")
    parser.add_argument('--longitude', type=str, required=True, help="Grid Longitude Value")
    parser.add_argument('--start_yr', type=int, default=1951, help="Start Year")
    parser.add_argument('--end_yr', type=int, default=2023, help="End Year")
    parser.add_argument('--variable', type=str, default='tmax', help="Variable to analyze")
    parser.add_argument('--file_dir', type=str, default='./', help="File directory for data")
    parser.add_argument('--plot_all', type = bool, default=False, help="Plot Full time series for gridpoint")
    parser.add_argument('--plot_yearly', type = bool, default=False, help="Plot Yearly time series for gridpoint")
    parser.add_argument('--plot_histograms', type = bool, default=True, help="Plot Histograms of yearly and daily frequencies")
    parser.add_argument('--plot_edf', type = bool, default=False, help="Plot Empirical Distribution functions for yearly and daily frequencies")
    parser.add_argument('--fit_gev', type = bool, default=False, help="Fit GEV to Daily Maxima and save parameters")
    parser.add_argument('--fit_gumbel', type = bool, default=False, help="Fit Gumbel to Daily Maxima and save parameters")
    parser.add_argument('--do_halves', type = bool, default=False, help="Split data in 2 halves, fit distributions to both and plot fits")

    args = parser.parse_args()
    lon_boundaries = np.linspace(67.5, 97.5, 31)
    lat_boundaries = np.linspace(7.5, 37.5, 31)
    lat_pos = np.digitize([args.latitude], lat_boundaries)
    lon_pos = np.digitize([args.longitude], lon_boundaries)
    latitude = lat_boundaries[lat_pos - 1]
    longitude = lon_boundaries[lon_pos - 1]
    data = get_grid_data(args, latitude, longitude)
    create_directories(args.name, latitude, longitude)
    
    yearly_max, yearly_min, yearly_mean = calculate_yearly_statistics(data)
    
    if args.plot_yearly:
        plot_trend(yearly_max, f"Largest Maximum Daily Temperatures from {args.start_yr} to {args.end_yr} at {args.name}", 
                'Value', f"./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/max_trend.png")
        plot_trend(yearly_min, f"Smallest Maximum Daily Temperatures from {args.start_yr} to {args.end_yr} at {args.name}", 
                'Value', f"./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/min_trend.png")
        plot_trend(yearly_mean, f"Mean Maximum Daily Temperatures from {args.start_yr} to {args.end_yr} at {args.name}", 
                'Value', f"./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/mean_trend.png")
    
    if args.plot_all:
        plot_full_data(data, f'./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/full_data.png')
    
    if args.plot_histograms:
        plot_histogram(data, "Histogram Plot of Daily Max Frequencies", f'./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/daily_max_histogram.png')
        plot_histogram(yearly_max, "Histogram Plot of Yearly Max Frequencies", f'./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/max_histogram.png')
        plot_histogram(yearly_min, "Histogram Plot of Smallest Yearly Max", f'./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/min_histogram.png')
        plot_histogram(yearly_mean, "Histogram Plot of Mean Yearly Max", f'./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/mean_histogram.png')
    
    if args.plot_edf:
        plot_edf(data, f'./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/edf_data.png')
        plot_edf(yearly_max, f'./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/edf_block_maxima.png')
        plot_edf(yearly_min, f'./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/edf_smallest_block_maxima.png')
        plot_edf(yearly_mean, f'./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/edf_mean_block_maxima.png')

    if args.fit_gev:
        params_mle_gev, ll_mle_gev = fit_distribution(yearly_max, scipy.stats.genextreme, 'gev')
        save_fit_parameters(args.name, latitude, longitude, params_mle_gev, ll_mle_gev, 'gev')
    
    if args.fit_gumbel:
        params_mle_gumbel, ll_mle_gumbel = fit_distribution(yearly_max, scipy.stats.gumbel_r, 'gumbel')
        save_fit_parameters(args.name, latitude, longitude, params_mle_gumbel, ll_mle_gumbel, 'gumbel')
    if args.fit_gumbel and args.fit_gev:
        plot_distribution_comparison(yearly_max, params_mle_gev, params_mle_gumbel, 
                                    f"./plots/{args.name}_{latitude:.2f}_{longitude:.2f}/gev_vs_gumbel_fit.png")
    
    fit_and_plot_half_data(yearly_max, args.name, latitude, longitude)

if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import geopandas as gpd
plt.switch_backend('Agg')
      
def process_location(lat, lon, data = None, data_type = 'tmax', data_name = 'Maxima'):
    directory_path = f"./trend_plots/{lat:.2f}_{lon:.2f}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if(max(data) > 53):
        print(f"Unlikely measurement of {max(data)} at coordinates {(lat, lon)}")
        return pd.Series()
    yearly_max = data.resample('Y').max()
    yearly_min = data.resample('Y').min()
    yearly_mean = data.resample('Y').mean()
    X_max = np.arange(len(yearly_max)).reshape(-1, 1)
    y_max = yearly_max.values
    model_max = LinearRegression().fit(X_max, y_max)
    trend = model_max.predict(X_max)
    plt.plot(yearly_max.index, yearly_max, label='Time Series', color = 'blue')
    plt.plot(yearly_max.index, trend, 'r--', label='Trend Line')
    plt.title(f"Largest Daily {data_name} from 1951 to 2019 at coordinates {lat, lon}")
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.savefig(f"./trend_plots/{lat:.2f}_{lon:.2f}/{data_type}_max_trend.png")
    plt.clf()

    X_min = np.arange(len(yearly_min)).reshape(-1, 1)
    y_min = yearly_min.values
    model_min = LinearRegression().fit(X_min, y_min)
    trend = model_min.predict(X_min)
    plt.plot(yearly_min.index, yearly_min, label='Time Series', color = 'blue')
    plt.plot(yearly_min.index, trend, 'r--', label='Trend Line')
    plt.title(f"Smallest Daily {data_name} from 1951 to 2019 at coordinates {lat, lon}")
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.savefig(f"./trend_plots/{lat:.2f}_{lon:.2f}/{data_type}_min_trend.png")
    plt.clf()

    X_mean = np.arange(len(yearly_mean)).reshape(-1, 1)
    y_mean = yearly_mean.values
    model_mean = LinearRegression().fit(X_mean, y_mean)
    trend = model_mean.predict(X_mean)
    plt.plot(yearly_mean.index, yearly_mean, label='Time Series', color = 'blue')
    plt.plot(yearly_mean.index, trend, 'r--', label='Trend Line')
    plt.title(f"Mean Daily {data_name} from 1951 to 2019 at coordinates {lat, lon}")
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.savefig(f"./trend_plots/{lat:.2f}_{lon:.2f}/{data_type}_mean_trend.png")
    plt.clf()

    return pd.Series({
        'latitude': lat,
        'longitude': lon,
        'max_slope': model_max.coef_[0],
        'min_slope': model_min.coef_[0],
        'mean_slope': model_mean.coef_[0],
        'max_intercept': model_max.intercept_,
        'min_intercept': model_min.intercept_,
        'mean_intercept': model_mean.intercept_,
    })
    
def plot_trends(lats, lons, path="./per_grid_data/tmax_data.csv", data_type = "tmax"):
    data_name = 'Maxima' if data_type == 'tmax' else 'Minima'
    data = pd.read_csv(path)
    results_df = pd.DataFrame([], columns = ['latitude', 'longitude', 'max_slope', 'min_slope', 'mean_slope', 'max_intercept', 'min_intercept', 'mean_intercept'])
    for lat in lats:
        lat_data = data[data['lat'] == lat]
        for lon in lons:
            lon_data = lat_data[lat_data['lon'] == lon]
            relevant_data = lon_data.drop(columns = ['lat', 'lon'])
            relevant_data['time'] = pd.to_datetime(relevant_data['time'])
            relevant_data.set_index('time', inplace=True)
            series = relevant_data[relevant_data.columns[0]]
            if(max(series)>=54):
                print(f"Unlikely measurement of {max(series)} at coordinates {(lat, lon)}")
                print(series[series > 53].value_counts())
                continue
            result = process_location(lat, lon, series, data_type)
            if not result.empty:
                results_df = pd.concat([results_df, result.to_frame().T], ignore_index=True)
    
    results_df.to_csv(f"./max_min_mean_trend_heatmaps/{data_type}_trend_results.csv", index = False)
    india = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    india = india[india.name == "India"]

    geometry = gpd.points_from_xy(results_df['longitude'], results_df['latitude'])
    grid_gdf = gpd.GeoDataFrame(results_df, geometry=geometry)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    india.boundary.plot(ax=ax)
    grid_gdf.plot(column='max_slope', ax=ax, legend=True, cmap='coolwarm', markersize=50)
    plt.title(f'Gradient Heatmap of Slopes for Largest Daily {data_name} over India')
    plt.savefig(f"./max_min_mean_trend_heatmaps/{data_type}_max_slopes_heatmap.png")
    plt.clf()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    india.boundary.plot(ax=ax)
    grid_gdf.plot(column='min_slope', ax=ax, legend=True, cmap='coolwarm', markersize=50)
    plt.title(f'Gradient Heatmap of Slopes for Smallest Daily {data_name} over India')
    plt.savefig(f"./max_min_mean_trend_heatmaps/{data_type}_min_slopes_heatmap.png")
    plt.clf()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    india.boundary.plot(ax=ax)
    grid_gdf.plot(column='mean_slope', ax=ax, legend=True, cmap='coolwarm', markersize=50)
    plt.title(f'Gradient Heatmap of Slopes for Mean Daily {data_name} over India')
    plt.savefig(f"./max_min_mean_trend_heatmaps/{data_type}_mean_slopes_heatmap.png")
    plt.clf()

def main():
    lons = np.linspace(67.5, 97.5, 31)
    lats = np.linspace(7.5, 37.5, 31)
    plot_trends(lats, lons)
    plot_trends(lats, lons, 'per_grid_data/tmin_data.csv', 'tmin')
    
if __name__ == "__main__":
    main()
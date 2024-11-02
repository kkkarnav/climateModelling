import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import geopandas as gpd
plt.switch_backend('Agg')

def calculate_exceedances_and_spells(data, percentiles=[95]):
    print(percentiles)
    results = {}
    for x in percentiles:
        threshold = np.percentile(data, x)
        above_threshold = data > threshold
        
        # Series with continuous periods above the threshold
        spells = (above_threshold != above_threshold.shift()).cumsum() #Indexing every spell
        spells_above_threshold = spells[above_threshold] #Keeping only the hot spell
        
        for year, group in above_threshold.groupby(above_threshold.index.year):
            if year not in results.keys():
                results[year] = {}
            num_days_above = group.sum()
            if num_days_above > 0:
                yearly_spells = spells_above_threshold[spells_above_threshold.index.year == year]
                continuous_spells = yearly_spells.value_counts()
                mean_spell_length = continuous_spells.mean()
            else:
                mean_spell_length = 0
            
            results[year][f'num_days_above_{x}'] = num_days_above
            results[year][f'mean_spell_length_above_{x}'] = mean_spell_length
        print(results)
    print(results)
    return pd.DataFrame(results).T

def get_exceedences(data, lat, lon, percentiles):
    directory_path = f"./trend_plots/{lat:.2f}_{lon:.2f}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if max(data) > 53:
        print(f"Unlikely measurement of {max(data)} at coordinates {(lat, lon)}")
        return pd.Series()
    
    results = {"latitude": lat, "longitude": lon}
    print(percentiles)
    exceedances_and_spells = calculate_exceedances_and_spells(data, percentiles)
    print(exceedances_and_spells.columns)
    X = exceedances_and_spells.index.values.reshape(-1, 1)
    exceedances_and_spells.to_csv(f"./trend_plots/{lat:.2f}_{lon:.2f}/exceedances_and_spells.csv", index=False)
    for x in percentiles:
        y_days = exceedances_and_spells[f'num_days_above_{x}'].values
        y_spells = exceedances_and_spells[f'mean_spell_length_above_{x}'].values

        model_days = LinearRegression().fit(X, y_days)
        model_spells = LinearRegression().fit(X, y_spells)

        results[f'{x}_days_slope'] = model_days.coef_[0]
        results[f'{x}_spells_slope'] = model_spells.coef_[0]
        results[f'{x}_days_intercept'] = model_days.intercept_
        results[f'{x}_spells_intercept'] = model_spells.intercept_
    
    return pd.Series(results)

def plot_spell_lengths(lats, lons, percentiles, data_type = 'tmax'):
    data = pd.read_csv('./dataset/heat_main/main_tmax_panel.csv')
    data.columns = data.columns.str.lower()
    data.rename({"long": "lon", "date": "time"}, inplace=True, axis=1)
    
    columns = [f'avg_spell_above_{x}' for x in percentiles]
    columns.append('latitude')
    columns.append('longitude')
    results_df = pd.DataFrame([], columns = columns)
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
            result = get_exceedences(series, lat, lon, percentiles)
            if not result.empty:
                results_df = pd.concat([results_df, result.to_frame().T], ignore_index=True)
    results_df.to_csv("./dataset/spell_lengths/spell_length_data.csv", index = False)
    
    india = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    india = india[india.name == "India"]

    geometry = gpd.points_from_xy(results_df['longitude'], results_df['latitude'])
    grid_gdf = gpd.GeoDataFrame(results_df, geometry=geometry)
    
    for x in percentiles:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        india.boundary.plot(ax=axes[0])
        grid_gdf.plot(column=f'{x}_days_intercept', ax=axes[0], legend=True, cmap='coolwarm', markersize=50)
        axes[0].set_title(f'Intercept of number of days above {x}th percentile')

        india.boundary.plot(ax=axes[1])
        grid_gdf.plot(column=f'{x}_days_slope', ax=axes[1], legend=True, cmap='coolwarm', markersize=50)
        axes[1].set_title(f'Trend of number of days above {x}th percentile')

        plt.tight_layout()
        plt.savefig(f"./dataset/spell_lengths/num_days_over_{x}_trend.png")
        plt.clf()
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        india.boundary.plot(ax=axes[0])
        grid_gdf.plot(column=f'{x}_days_intercept', ax=axes[0], legend=True, cmap='coolwarm', markersize=50)
        axes[0].set_title(f'Intercept of average spell length above {x}th percentile')

        india.boundary.plot(ax=axes[1])
        grid_gdf.plot(column=f'{x}_days_slope', ax=axes[1], legend=True, cmap='coolwarm', markersize=50)
        axes[1].set_title(f'Trend of average spell length above {x}th percentile')

        plt.tight_layout()
        plt.savefig(f"./dataset/spell_lengths/spell_length_above_{x}_trend.png")
        plt.clf()
        
def main():
    lons = np.linspace(73.5, 97.5, 31)
    lats = np.linspace(8.5, 37.5, 31)
    plot_spell_lengths(lats, lons, [80, 90, 95])
    
if __name__ == "__main__":
    main()

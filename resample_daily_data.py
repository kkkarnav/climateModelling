import pandas as pd
import numpy as np

def calculate_daily_stats(max_df, min_df, threshold=54):
    merged_df = pd.merge(max_df, min_df, on=['lat', 'lon', 'time', 'date'])
    merged_df = merged_df[(merged_df['tmax'] <= threshold) & (merged_df['tmin'] <= threshold)]
    
    daily_max = merged_df.groupby(['lat', 'lon', 'date'])['tmax'].max().reset_index()
    daily_min = merged_df.groupby(['lat', 'lon', 'date'])['tmin'].min().reset_index()
    daily_mean_max = merged_df.groupby(['lat', 'lon', 'date'])['tmax'].mean().reset_index().rename(columns={'tmax': 'tmax_mean'})
    daily_mean_min = merged_df.groupby(['lat', 'lon', 'date'])['tmin'].mean().reset_index().rename(columns={'tmin': 'tmin_mean'})
    
    daily_stats = pd.merge(daily_max, daily_min, on=['lat', 'lon', 'date'])
    daily_stats = pd.merge(daily_stats, daily_mean_max, on=['lat', 'lon', 'date'])
    daily_stats = pd.merge(daily_stats, daily_mean_min, on=['lat', 'lon', 'date'])
    
    daily_stats['tmean'] = daily_stats[['tmax_mean', 'tmin_mean']].mean(axis=1)
    
    daily_stats = daily_stats.drop(columns=['tmax_mean', 'tmin_mean'])
    
    return daily_stats

def area_weighted_resample(data, old_res=0.5, new_res=1.0):
    new_lat = np.arange(data['lat'].min(), data['lat'].max() + new_res, new_res)
    new_lon = np.arange(data['lon'].min(), data['lon'].max() + new_res, new_res)
    
    resampled_data = []

    for date in data['date'].unique():
        daily_data = data[data['date'] == date]
        
        for nx in new_lat:
            for ny in new_lon:
                # Select all cells that fall within the new grid cell
                cells = daily_data[
                    (daily_data['lat'] >= nx - old_res/2) & (daily_data['lat'] < nx + old_res/2) &
                    (daily_data['lon'] >= ny - old_res/2) & (daily_data['lon'] < ny + old_res/2)
                ]
                
                if not cells.empty:
                    # Calculate the area-weighted average
                    avg_max_temp = cells['tmax'].mean()
                    avg_min_temp = cells['tmin'].mean()
                    avg_mean_temp = cells['tmean'].mean()
                    resampled_data.append([nx, ny, date, avg_max_temp, avg_min_temp, avg_mean_temp])
    
    resampled_df = pd.DataFrame(resampled_data, columns=['lat', 'lon', 'time', 'tmax', 'tmin', 'tmean'])
    return resampled_df

hourly_max = pd.read_csv('realtime_tmax_data.csv', parse_dates=['time'])
hourly_max['date'] = hourly_max['time'].dt.date

hourly_min = pd.read_csv('realtime_tmin_data.csv', parse_dates=['time'])
hourly_min['date'] = hourly_min['time'].dt.date

daily_stats = calculate_daily_stats(hourly_max, hourly_min)

resampled_stats = area_weighted_resample(daily_stats)
resampled_stats.to_csv('tmax_tmin_tmean_daily_2020_1_by_1.csv', index=False)

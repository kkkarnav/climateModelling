import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_temperature_comparison(local_data, global_predictions):
    """
    Precise temperature comparison analysis with specific time alignment.
    
    Parameters:
    -----------
    local_data : pandas.DataFrame
        Local temperature dataset 
    global_predictions : xarray.Dataset
        Global temperature predictions dataset
    
    Returns:
    --------
    pd.DataFrame: Detailed temperature comparison metrics
    """
    # Ensure date is in datetime format
    local_data['date'] = pd.to_datetime(local_data['date'])
    
    # Filter local data from 2nd Jan 2022
    local_data_filtered = local_data[local_data['date'] >= pd.Timestamp('2022-01-02')]
    
    # save the filtered data
    local_data_filtered.to_csv("dataset/2-1-2022_filtered_imd.csv", index=False)

    # Load global predictions from nc file to xarray
    global_predictions = xr.open_dataset(global_predictions)

    print(global_predictions['2m_temperature'])
    
    
    # Prepare to store comparison results
    comparison_results = []
    
    # Specific time indices for comparison (0, 1, 3, 4, 5)
    time_indices = [0, 1, 3, 4, 5]
    
    for idx, time_index in enumerate(time_indices):
        # Extract global temperature for this time step
        global_temp_slice = global_predictions['2m_temperature'].isel(time=time_index) - 273.15
        
        # Find corresponding local data for this date
        local_subset = local_data.iloc[idx:idx+1]
        
        try:
            # Find closest grid point in global predictions
            closest_temp = global_temp_slice.sel(
                lat=local_subset['LAT'].values[0], 
                lon=local_subset['LONG'].values[0], 
                method='nearest'
            ).values
            
            comparison_results.append({
                'date': local_subset['date'].values[0],
                'local_temp': local_subset['TEMP'].values[0],
                'global_pred_temp': closest_temp,
                'lat': local_subset['LAT'].values[0],
                'lon': local_subset['LONG'].values[0],
                'temp_difference': local_subset['TEMP'].values[0] - closest_temp
            })
        except Exception as e:
            print(f"Error processing data for {local_subset['date'].values[0]}: {e}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Local Temperatures
    plt.subplot(131)
    scatter1 = plt.scatter(
        comparison_df['lon'], 
        comparison_df['lat'], 
        c=comparison_df['local_temp'], 
        cmap='hot'
    )
    plt.colorbar(scatter1, label='Local Temperature (°C)')
    plt.title('Local Temperatures')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Global Predictions
    plt.subplot(132)
    scatter2 = plt.scatter(
        comparison_df['lon'], 
        comparison_df['lat'], 
        c=comparison_df['global_pred_temp'], 
        cmap='hot'
    )
    plt.colorbar(scatter2, label='Global Predicted Temperature (°C)')
    plt.title('Global Temperature Predictions')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Temperature Difference
    plt.subplot(133)
    scatter3 = plt.scatter(
        comparison_df['lon'], 
        comparison_df['lat'], 
        c=comparison_df['temp_difference'], 
        cmap='coolwarm', 
        vmin=-5, 
        vmax=5
    )
    plt.colorbar(scatter3, label='Temperature Difference (°C)')
    plt.title('Local vs Global Temperature Difference')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.show()
    
    # Comprehensive Statistical Analysis
    print("\nTemperature Comparison Analysis:")
    
    # Descriptive Statistics
    print("\nDescriptive Statistics:")
    desc_stats = comparison_df[['local_temp', 'global_pred_temp', 'temp_difference']].describe()
    print(desc_stats)
    
    # Error Metrics
    mae = np.mean(np.abs(comparison_df['temp_difference']))
    rmse = np.sqrt(np.mean(comparison_df['temp_difference']**2))
    print(f"\nError Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f}°C")
    
    # Correlation Analysis
    correlation = comparison_df['local_temp'].corr(comparison_df['global_pred_temp'])
    print(f"\nCorrelation between Local and Global Temperatures: {correlation:.4f}")
    
    return comparison_df
# Usage
# Assuming 'dataset' is your local temperature DataFrame and 'path/to/your/file.nc' is the NetCDF file path
dataset = pd.read_csv("dataset/heat_main2/heat_main/main_tmax_panel.csv")
result = process_temperature_comparison(dataset, './dataset/heat_main2/predictions_old.nc')
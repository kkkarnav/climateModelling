import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from concurrent.futures import ProcessPoolExecutor

def process_isobar_xarray(ds_isobar, lat_edges, lon_edges):
    results = []

    # Create 1D arrays of latitude and longitude bins
    lat_bins = xr.DataArray(
        np.clip(np.digitize(ds_isobar['latitude'], lat_edges) - 1, 0, len(lat_edges) - 2),
        dims='latitude',
        coords={'latitude': ds_isobar['latitude']}
    )
    lon_bins = xr.DataArray(
        np.clip(np.digitize(ds_isobar['longitude'], lon_edges) - 1, 0, len(lon_edges) - 2),
        dims='longitude',
        coords={'longitude': ds_isobar['longitude']}
    )

    # Resample time dimension to daily max for each isobaric level
    ds_daily_max = ds_isobar.resample(time='1D').max(dim='time')

    # Iterate over unique combinations of lat_bins and lon_bins
    for lat_bin in np.unique(lat_bins):
        for lon_bin in np.unique(lon_bins):
            cell_data = ds_daily_max.where((lat_bins == lat_bin) & (lon_bins == lon_bin), drop=True)

            if cell_data.sizes['latitude'] > 0 and cell_data.sizes['longitude'] > 0:
                lat_min = lat_edges[lat_bin]
                lat_max = lat_edges[min(lat_bin + 1, len(lat_edges) - 1)]
                lon_min = lon_edges[lon_bin]
                lon_max = lon_edges[min(lon_bin + 1, len(lon_edges) - 1)]

                daily_max_temps = cell_data['t'].max(dim=['latitude', 'longitude'])

                avg_of_max_temps = daily_max_temps.mean().item()

                for time, max_temp in zip(cell_data['time'], daily_max_temps):
                    results.append({
                        'date': str(time.values)[:10],  # Only keep the date part
                        'lat_min': lat_min,
                        'lat_max': lat_max,
                        'lon_min': lon_min,
                        'lon_max': lon_max,
                        'max_value': max_temp.item(),
                        'avg_of_max_temps': avg_of_max_temps
                    })

    return results


def process_grib2_file(file, output_dir, isobar_values, lat_edges, lon_edges):
    print(f"Processing file: {file}")

    try:
        ds = xr.open_dataset(file, engine='cfgrib')
        print(f"Dataset opened successfully. Dimensions: {ds.dims}")

        for isobar_value in isobar_values:
            print(f"Processing isobar value: {isobar_value}")
            # Select the correct isobaric level using xarray's .sel() method
            ds_isobar = ds.sel(isobaricInhPa=isobar_value)
            print(f"Isobar level selected. Dimensions: {ds_isobar.dims}")

            # Process this isobaric level and perform the reduction
            results = process_isobar_xarray(ds_isobar, lat_edges, lon_edges)
            print(f"Processed isobar level. Number of results: {len(results)}")

            df_results = pd.DataFrame(results)

            year = os.path.basename(file).split('.')[0]

            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f'IMDAA_tmax_0.25_{isobar_value}_{year}.csv')
            df_results.to_csv(output_file, index=False)
            print(f"Saved results to {output_file}")

    except Exception as e:
        print(f"Error processing file {file}: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        if 'ds' in locals():
            ds.close()


def process_grib2_files_parallel(input_dir, file_pattern, output_dir, isobar_values, lat_edges, lon_edges):
    files = glob.glob(os.path.join(input_dir, file_pattern))
    print(f"Found {len(files)} files to process")

    # Use a ProcessPoolExecutor to parallelize the file processing across 20 cores
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(process_grib2_file, file, output_dir, isobar_values, lat_edges, lon_edges)
            for file in files
        ]

        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in concurrent processing: {e}")

    print("All files processed.")


directory_path = 'IMDAA' 
file_pattern = '*.grb2' 
output_dir = "./IMDAA_csv" 
isobar_values = [1000, 850] 

lons = np.arange(67.5, 97.75, 0.25)
lats = np.arange(7.5, 37.75, 0.25)
process_grib2_files_parallel(directory_path, file_pattern, output_dir, isobar_values, lats, lons)

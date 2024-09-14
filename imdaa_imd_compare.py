import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_and_process_file(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'lat_min': 'lat', 'lon_min': 'lon'})
    df = df.drop(columns=['lat_max', 'lon_max'])
    # print(f"dataframe after loading: {df.head(100)}")
    return df

def process_grid_chunk(chunk, reference_grid):
    def assign_grid(value):
        return np.floor(value - 0.5) + 0.5

    # Assign grid values
    chunk['lat_grid'] = chunk['lat'].apply(assign_grid)
    chunk['lon_grid'] = chunk['lon'].apply(assign_grid)
    
    # Group and aggregate
    processed = chunk.groupby(['date', 'lat_grid', 'lon_grid']).agg({
        'max_value': 'max',
        'avg_of_max_temps': 'mean'
    }).reset_index()
    
    # Rename columns to match the desired output
    processed = processed.rename(columns={'lat_grid': 'lat', 'lon_grid': 'lon'})
    # if(processed['max_value'].isna().sum() !=0):
    #     print("Before merging")
    #     print(processed.head(100))
    #     print(processed.shape)
    #     print(processed[processed['max_value'].isna()])
    #     print(processed[processed['max_value'].isna()]['max_value'])
    #     print(processed['avg_of_max_temps'].isna().sum())
    # Merge with reference grid
    processed = pd.merge(processed, reference_grid, on=['lat', 'lon'], how='inner')
    # if(processed['max_value'].isna().sum() !=0):
    #     print("After merging")
    #     print(processed.head(100))
    #     print(processed.shape)
    #     print(processed[processed['max_value'].isna()])
    #     print(processed[processed['max_value'].isna()]['max_value'])
    #     print(processed['avg_of_max_temps'].isna().sum())

    return processed

def load_and_process_first_dataframe(folder_path, file_pattern, reference_df, num_workers=20):
    all_files = glob.glob(os.path.join(folder_path, file_pattern))
    reference_grid = reference_df[['lat', 'lon']].drop_duplicates()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Load files in parallel
        future_to_df = {executor.submit(load_and_process_file, file): file for file in all_files}
        
        dfs = []
        for future in as_completed(future_to_df):
            dfs.append(future.result())
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    # print(f"dataframe after concatenating: {df.head(100)}")
    
    # Split the dataframe into chunks for parallel processing
    chunks = np.array_split(df, num_workers)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Process grid in parallel
        future_to_chunk = {executor.submit(process_grid_chunk, chunk, reference_grid): i for i, chunk in enumerate(chunks)}
        
        processed_chunks = []
        for future in as_completed(future_to_chunk):
            processed_chunks.append(future.result())
    
    # Concatenate all processed chunks
    return pd.concat(processed_chunks, ignore_index=True)

def load_second_csv(file_path):
    df = pd.read_csv(file_path)
    df = df[df['tmax'] < 55]
    df['date'] = pd.to_datetime(df['time'])
    return df

def align_data(df1, df2):
    # print(df1.head(100))
    # print(df2.head(100))
    return pd.merge(df1, df2, on=['date', 'lat', 'lon'], how='inner')

def calculate_metrics(df:pd.DataFrame):
    # print(df.head(100))
    # print(df.shape)
    # print(df[df['max_value'].isna()])
    # print(df[df['max_value'].isna()]['max_value'])
    # print(df['avg_of_max_temps'].isna().sum())
    # print(df['tmax'].isna().sum())
    df.dropna(inplace=True)
    df['max_value'] = df['max_value'] - 273.15
    df['avg_of_max_temps'] = df['avg_of_max_temps'] - 273.15
    corr_max, _ = pearsonr(df['max_value'], df['tmax'])
    corr_avg, _ = pearsonr(df['avg_of_max_temps'], df['tmax'])
    mse_max = mean_squared_error(df['max_value'], df['tmax'])
    mse_avg = mean_squared_error(df['avg_of_max_temps'], df['tmax'])
    return corr_max, corr_avg, mse_max, mse_avg

def main(folder_path, file_pattern, second_file_path, num_workers=20):
    # Load second CSV first (we need it for reference grid)
    df2 = load_second_csv(second_file_path)
    
    # Load and process first dataframe
    df1 = load_and_process_first_dataframe(folder_path, file_pattern, df2, num_workers)
    
    # Align data
    combined_df = align_data(df1, df2)
    # print(combined_df.head(100))
    # print(combined_df.shape)
    # print(combined_df[combined_df['max_value'].isna()])
    # print(combined_df[combined_df['max_value'].isna()]['max_value'])
    # print(combined_df['avg_of_max_temps'].isna().sum())
    # print(combined_df['tmax'].isna().sum())
    # Calculate metrics
    corr_max, corr_avg, mse_max, mse_avg = calculate_metrics(combined_df)
    
    print(f"Correlation coefficient (max_value vs tmax): {corr_max}")
    print(f"Correlation coefficient (avg_of_max_temps vs tmax): {corr_avg}")
    print(f"MSE (max_value vs tmax): {mse_max}")
    print(f"MSE (avg_of_max_temps vs tmax): {mse_avg}")

if __name__ == "__main__":
    main("IMDAA_csv", "*_850_*.csv", "per_grid_data/tmax_summer.csv", 20)
    main("IMDAA_csv", "*_1000_*.csv", "per_grid_data/tmax_summer.csv", 20)
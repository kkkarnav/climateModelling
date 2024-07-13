import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import warnings
import imdlib as imd

warnings.filterwarnings("ignore")
OUTPUT_DIR = "./dataset/heat_rt"


def grab_rt_heat_data(variable):
    # Data download parameters
    output_path = f'{OUTPUT_DIR}/{variable}'

    # Earliest available 0.5x0.5 data is 2015 onwards
    data = imd.get_real_data(variable, "2015-06-01", "2015-12-31", file_dir=output_path)
    df = data.get_xarray().to_dataframe().unstack(level=1).unstack(level=1)
    df.to_csv(f"{OUTPUT_DIR}/0.5_{variable}_2015.csv")

    # Download the rest of the 0.5x0.5 data
    for year in range(2016, 2024):
        start_day = f"{year}-01-01"
        end_day = f"{year}-12-31"

        # Call the API and dump to file
        data = imd.get_real_data(variable, start_day, end_day, file_dir=output_path)
        df = data.get_xarray().to_dataframe().unstack(level=1).unstack(level=1)
        df.to_csv(f"{OUTPUT_DIR}/{variable}/0.5_{variable}_{year}.csv")


def read_rt_heat_data(variable):
    df = pd.DataFrame()
    for year in range(2015, 2024):
        year_df = pd.read_csv(f"{OUTPUT_DIR}/{variable}/0.5_{variable}_{year}.csv", header=[1, 2])[1:] \
            .reset_index(drop=True) \
            .replace(99.9000015258789, -99)
        df = pd.concat([df, year_df])

    df = df.reset_index(drop=True)
    count = (df == -99).sum()
    df = df.drop(columns=count[count > 30].index)
    df = df.reset_index(drop=True).replace(-99, 26.98989)

    df.to_csv(f"{OUTPUT_DIR}/0.5_{variable}.csv")
    return df


def get_rt_data():
    # Get raw tmin and tmax data as GRD files from IMDlib
    grab_rt_heat_data("tmin")
    grab_rt_heat_data("tmax")

    # Read raw tmin and tmax data into csv format
    tmin, tmax = read_rt_heat_data("tmin"), read_rt_heat_data("tmax")
    
    print(tmin.head())
    print(tmax.head())


if __name__ == "__main__":
    get_rt_data()

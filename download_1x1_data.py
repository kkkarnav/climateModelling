import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import warnings
import imdlib as imd

warnings.filterwarnings("ignore")
OUTPUT_DIR = "./dataset/heat_main"


def grab_heat_data(variable):
    # Data download parameters
    output_path = f'{OUTPUT_DIR}'
    start_year = 1951
    end_year = 2023

    # Earliest available data is 1950 onwards
    data = imd.get_data(variable, start_year, end_year, fn_format="yearwise", file_dir=output_path)
    df = data.get_xarray().to_dataframe().unstack(level=1).unstack(level=1)
    df.to_csv(f"{OUTPUT_DIR}/main_{variable}.csv")

    print(df)


def read_heat_data(variable):
    df = pd.DataFrame()

    df = pd.read_csv(f"{OUTPUT_DIR}/main_{variable}.csv", header=[1, 2])[1:] \
        .reset_index(drop=True) \
        .replace(99.9000015258789, -99)

    count = (df == -99).sum()
    df = df.drop(columns=count[count > 40].index)
    df = df.reset_index(drop=True).replace(-99, 25.98989)

    return df


def get_main_data():
    # Get raw tmin and tmax data as GRD files from IMDlib
    grab_heat_data("tmin")
    grab_heat_data("tmax")

    # Read raw tmin and tmax data into csv format
    tmin, tmax = read_heat_data("tmin"), read_heat_data("tmax")
    
    print(tmin.head())
    print(tmax.head())


if __name__ == "__main__":
    get_main_data()

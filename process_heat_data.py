import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import warnings
import imdlib as imd

warnings.filterwarnings("ignore")
OUTPUT_DIR = "./dataset/heat_main"


def read_heat_data(variable):
    df = pd.DataFrame()

    df = pd.read_csv(f"{OUTPUT_DIR}/main_{variable}.csv", header=[1, 2])[1:] \
        .reset_index(drop=True) \
        .replace(99.9000015258789, -99)

    count = (df == -99).sum()
    # df = df.drop(columns=count[count > 40].index)
    df = df.reset_index(drop=True)  # .replace(-99, 25.98989)

    return df


def read_rt_heat_data(variable):
    df = pd.DataFrame()
    for year in range(2015, 2024):
        year_df = pd.read_csv(f"{OUTPUT_DIR}/{variable}/0.5_{variable}_{year}.csv", header=[1, 2])[1:] \
            .reset_index(drop=True) \
            .replace(99.9000015258789, -99)
        df = pd.concat([df, year_df])

    df = df.reset_index(drop=True)
    count = (df == -99).sum()
    # df = df.drop(columns=count[count > 30].index)
    df = df.reset_index(drop=True)  # .replace(-99, 26.98989)

    df.to_csv(f"{OUTPUT_DIR}/0.5_{variable}.csv")
    return df


def process_raw_data(dfmin, dfmax, prefix):
    dfmean = dfmin.copy()
    dfrange = dfmin.copy()

    for column in tqdm(range(1, len(dfmin.iloc[0]))):
        for row in range(len(dfmin)):
            dfmean.iloc[row, column] = (dfmax.iloc[row, column] + dfmin.iloc[row, column]) / 2
            dfrange.iloc[row, column] = dfmax.iloc[row, column] - dfmin.iloc[row, column]

    dfmean.to_csv(f"{OUTPUT_DIR}/{prefix}_tmean.csv")
    dfrange.to_csv(f"{OUTPUT_DIR}/{prefix}_dtr.csv")
    return dfmean, dfrange


def convert_to_format(df, variable, prefix):
    # Melt the df
    df = df.melt(id_vars=[("lat", "lon")], value_name="TEMP")
    df = df.rename(columns={("lat", "lon"): "date", "variable_0": "LAT", "variable_1": "LONG"})

    # Sort it
    df["LAT"] = df["LAT"].apply(lambda x: float(x))
    df["LONG"] = df["LONG"].apply(lambda x: float(x))
    df = df.sort_values(by=["LAT", "LONG", "date"]).reset_index(drop=True)

    # Shift the value column to second
    cols = list(df.columns)
    df = df[[cols[0], cols[-1]] + cols[1:-1]]

    pprint(df)

    df.to_csv(f"{OUTPUT_DIR}/{prefix}_{variable}_panel.csv", index=False)
    return df


def process_main_data():

    # Read raw tmin and tmax data into csv format
    tmin, tmax = read_heat_data("tmin"), read_heat_data("tmax")

    # Convert tmin and tmax to the mean temperature per day
    tmean, dtr = process_raw_data(tmin, tmax, "main")

    tmaxf = convert_to_format(tmax, "tmax", "main")
    tminf = convert_to_format(tmin, "tmin", "main")
    tmeanf = convert_to_format(tmean, "tmean", "main")
    dtrf = convert_to_format(dtr, "dtr", "main")

    print(tmaxf.head())
    print(dtrf.head())


def process_rt_data():

    # Read raw tmin and tmax data into csv format
    tmin, tmax = read_rt_heat_data("tmin"), read_rt_heat_data("tmax")

    # Convert tmin and tmax to the mean temperature per day
    tmean, dtr = process_raw_data(tmin, tmax, "rt")

    tmaxf = convert_to_format(tmax, "tmax", "rt")
    tminf = convert_to_format(tmin, "tmin", "rt")
    tmeanf = convert_to_format(tmean, "tmean", "rt")
    dtrf = convert_to_format(dtr, "dtr", "rt")
    
    print(tmaxf.head())
    print(dtrf.head())


if __name__ == "__main__":
    process_main_data()
    # process_rt_data()

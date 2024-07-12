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
    df = df.drop(columns=count[count > 40].index)
    df = df.reset_index(drop=True).replace(-99, 25.98989)

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
    df = df.drop(columns=count[count > 30].index)
    df = df.reset_index(drop=True).replace(-99, 26.98989)

    df.to_csv(f"{OUTPUT_DIR}/0.5_{variable}.csv")
    return df


def visualize_raw_data(df, label):
    # India annual mean
    df["year"] = pd.to_datetime(df.iloc[:, 0]).apply(lambda x: x.year)
    annual_means = df.iloc[:, 1:].groupby("year").mean()
    ax = annual_means.mean(axis=1).plot(color="firebrick")
    ax.set_title(f'Annual means of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India annual min and max
    ax = annual_means.max(axis=1).plot(color="red", alpha=0.5)
    ax = annual_means.min(axis=1).plot(color="steelblue", alpha=0.5)
    ax = annual_means.mean(axis=1).plot(color="firebrick")
    ax.set_title(f'Annual min and max of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India monthly mean
    df["month"] = pd.to_datetime(df.iloc[:, 0]).apply(lambda x: x.month)
    monthly_means = df.iloc[:, 1:].groupby(["year", "month"]).mean().iloc[-240:]
    ax = monthly_means.mean(axis=1).plot(color="firebrick")
    ax.set_title(f'Monthly means of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()

    # India daily mean
    ax = df.iloc[:, 1:-2].mean(axis=1).iloc[-1825:].plot(color="firebrick")
    ax.set_xticks(df.iloc[-1825:].index[::366])
    ax.set_xticklabels(df.iloc[-1825:].iloc[:, 0][::366].apply(lambda x: x[:4]))
    ax.set_title(f'Daily means of {label} temperature across India', weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean temp')
    plt.show()


def viz_main_data():

    # Read raw tmin and tmax data into csv format
    tmin, tmax = read_heat_data("tmin"), read_heat_data("tmax")

    # Read the processed tmean data
    # tmean = pd.read_csv(f"{OUTPUT_DIR}/main_tmean.csv", header=[0, 1], index_col=0)
    # dtr = pd.read_csv(f"{OUTPUT_DIR}/main_dtr.csv", header=[0, 1], index_col=0)

    # Visualize temporal trends in min and max data
    # visualize_raw_data(tmin, "min.")
    visualize_raw_data(tmax, "max.")
    # visualize_raw_data(tmean, "mean")
    # visualize_raw_data(dtr, "range of")


def viz_rt_data():

    # Read raw tmin and tmax data into csv format
    tmin, tmax = read_rt_heat_data("tmin"), read_rt_heat_data("tmax")

    # Read the processed tmean data
    # tmean = pd.read_csv(f"{OUTPUT_DIR}/0.5_tmean.csv", header=[0, 1], index_col=0)
    # dtr = pd.read_csv(f"{OUTPUT_DIR}/0.5_dtr.csv", header=[0, 1], index_col=0)

    # Visualize temporal trends in min and max data
    # visualize_raw_data(tmin, "min.")
    visualize_raw_data(tmax, "max.")
    # visualize_raw_data(tmean, "mean")
    # visualize_raw_data(dtr, "range of")


if __name__ == "__main__":
    viz_main_data()
    # viz_rt_data()
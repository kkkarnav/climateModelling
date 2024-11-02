import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
from pprint import pprint
from tqdm import tqdm
import warnings
import geopandas as gpd
from shapely.geometry import box
import matplotlib
import glob
from PIL import Image    
import matplotlib.animation as animation

matplotlib.use('Agg')
tqdm.pandas()
warnings.filterwarnings("ignore")

DATA_DIR = "./dataset/heat_main"
OUTPUT_DIR = "./images/daily_plots"


def read_panel_heat_data(variable):
    df = pd.read_csv(f"{DATA_DIR}/main_{variable}_panel.csv").replace(-99, None)

    return df


def plot_regions(tseries):
    
    '''wc_tseries = tseries[
        (tseries["LAT"] <= 23.5) &
        (tseries["LONG"] <= 77.5) &
        (tseries["LAT"] + tseries["LONG"] < 95) &
        ~((tseries["LAT"] >= 13.5) & (tseries["LAT"] <= 17.5) & (tseries["LONG"] >= 76.5)) &
        ~((tseries["LAT"] >= 16.5) & (tseries["LAT"] <= 19.5) & (tseries["LONG"] >= 75.5))
    ]
    ec_tseries = tseries[
        (tseries["LAT"] <= 22.5) &
        (tseries["LONG"] >= 77.5) &
        ~((tseries["LAT"] >= 9.5) & (tseries["LONG"] < 78.5)) &
        ~((tseries["LAT"] >= 14.5) & (tseries["LONG"] < 79.5)) &
        ~((tseries["LAT"] >= 17.5) & (tseries["LONG"] < 80.5)) &
        ~((tseries["LAT"] >= 18.5) & (tseries["LONG"] < 81.5)) &
        ~((tseries["LAT"] >= 19.5) & (tseries["LONG"] < 82.5)) &
        ~((tseries["LAT"] >= 20.5) & (tseries["LONG"] < 83.5)) &
        ~((tseries["LAT"] >= 21.5) & (tseries["LONG"] < 84.5)) &
        ~((tseries["LAT"] >= 22.5) & (tseries["LONG"] < 85.5)) &
        ~((tseries["LAT"] >= 20.5) & (tseries["LONG"] >= 91.5))
    ]
    ne_tseries = tseries[
        (tseries["LAT"] >= 16.5) & (tseries["LONG"] >= 89.5) |
        (tseries["LAT"] >= 22.5) & (tseries["LONG"] >= 83.5)
    ]
    wh_tseries = tseries[
        (tseries["LAT"] >= 30.5)
    ]
    ip_tseries = tseries[
        (tseries["LAT"] <= 21.5) 
        &
        ~((tseries["LAT"] <= 23.5) &
        (tseries["LONG"] <= 77.5) &
        (tseries["LAT"] + tseries["LONG"] < 95) &
        ~((tseries["LAT"] >= 13.5) & (tseries["LAT"] <= 17.5) & (tseries["LONG"] >= 76.5)) &
        ~((tseries["LAT"] >= 16.5) & (tseries["LAT"] <= 19.5) & (tseries["LONG"] >= 75.5)))
        &
        ~((tseries["LAT"] <= 22.5) &
        (tseries["LONG"] >= 77.5) &
        ~((tseries["LAT"] >= 9.5) & (tseries["LONG"] < 78.5)) &
        ~((tseries["LAT"] >= 14.5) & (tseries["LONG"] < 79.5)) &
        ~((tseries["LAT"] >= 17.5) & (tseries["LONG"] < 80.5)) &
        ~((tseries["LAT"] >= 18.5) & (tseries["LONG"] < 81.5)) &
        ~((tseries["LAT"] >= 19.5) & (tseries["LONG"] < 82.5)) &
        ~((tseries["LAT"] >= 20.5) & (tseries["LONG"] < 83.5)) &
        ~((tseries["LAT"] >= 21.5) & (tseries["LONG"] < 84.5)) &
        ~((tseries["LAT"] >= 22.5) & (tseries["LONG"] < 85.5)) &
        ~((tseries["LAT"] >= 20.5) & (tseries["LONG"] >= 91.5)))
    ]
    nc_tseries = tseries[
        (tseries["LAT"] >= 21.5) & (tseries["LAT"] <= 30.5) & (tseries["LONG"] <= 83.5) &
        ~((tseries["LAT"] >= 21.5) & (tseries["LONG"] <= 73.5)) &
        ~((tseries["LAT"] >= 23.5) & (tseries["LONG"] <= 74.5)) &
        ~((tseries["LAT"] >= 24.5) & (tseries["LONG"] <= 75.5)) &
        ~((tseries["LAT"] >= 25.5) & (tseries["LONG"] <= 76.5)) &
        ~((tseries["LAT"] >= 26.5) & (tseries["LONG"] <= 77.5)) &
        ~((tseries["LAT"] >= 27.5) & (tseries["LONG"] <= 78.5)) &
        ~((tseries["LAT"] >= 28.5) & (tseries["LONG"] <= 79.5)) &
        ~((tseries["LAT"] >= 29.5) & (tseries["LONG"] <= 80.5))
    ]'''
    nw_tseries = tseries[
        (tseries["LAT"] >= 21.5) & (tseries["LAT"] <= 30.5) & (tseries["LONG"] <= 83.5) &
        ~(~((tseries["LAT"] >= 21.5) & (tseries["LONG"] <= 73.5)) &
        ~((tseries["LAT"] >= 23.5) & (tseries["LONG"] <= 74.5)) &
        ~((tseries["LAT"] >= 24.5) & (tseries["LONG"] <= 75.5)) &
        ~((tseries["LAT"] >= 25.5) & (tseries["LONG"] <= 76.5)) &
        ~((tseries["LAT"] >= 26.5) & (tseries["LONG"] <= 77.5)) &
        ~((tseries["LAT"] >= 27.5) & (tseries["LONG"] <= 78.5)) &
        ~((tseries["LAT"] >= 28.5) & (tseries["LONG"] <= 79.5)) &
        ~((tseries["LAT"] >= 29.5) & (tseries["LONG"] <= 80.5)))
    ]
    
    subset_tseries = nw_tseries.loc[nw_tseries["date"] == "2006-01-01", :]
    subset_tseries['geometry'] = subset_tseries.progress_apply(lambda row: box(row['LONG'], row['LAT'], row['LONG'] + 1, row['LAT'] + 1), axis=1)
    subset_tseries['TEMP'] = subset_tseries['TEMP'].apply(lambda x: float(x) if x else None)
    
    gdf = gpd.GeoDataFrame(subset_tseries, geometry='geometry')
    
    gdf.plot(column='TEMP', cmap='coolwarm', vmin=5, vmax=50, legend=True)
    plt.title(f'Daily max Temperatures for North West on 2006-01-01')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f'{OUTPUT_DIR}/nw.png')
    plt.clf()


def mark_regions(tseries):
    
    conditions = [
        # West Coast
        ((tseries["LAT"] <= 23.5) &
        (tseries["LONG"] <= 77.5) &
        (tseries["LAT"] + tseries["LONG"] < 95) &
        ~((tseries["LAT"] >= 13.5) & (tseries["LAT"] <= 17.5) & (tseries["LONG"] >= 76.5)) &
        ~((tseries["LAT"] >= 16.5) & (tseries["LAT"] <= 19.5) & (tseries["LONG"] >= 75.5))),
        # East Coast
        ((tseries["LAT"] <= 22.5) &
        (tseries["LONG"] >= 77.5) &
        ~((tseries["LAT"] >= 9.5) & (tseries["LONG"] < 78.5)) &
        ~((tseries["LAT"] >= 14.5) & (tseries["LONG"] < 79.5)) &
        ~((tseries["LAT"] >= 17.5) & (tseries["LONG"] < 80.5)) &
        ~((tseries["LAT"] >= 18.5) & (tseries["LONG"] < 81.5)) &
        ~((tseries["LAT"] >= 19.5) & (tseries["LONG"] < 82.5)) &
        ~((tseries["LAT"] >= 20.5) & (tseries["LONG"] < 83.5)) &
        ~((tseries["LAT"] >= 21.5) & (tseries["LONG"] < 84.5)) &
        ~((tseries["LAT"] >= 22.5) & (tseries["LONG"] < 85.5)) &
        ~((tseries["LAT"] >= 20.5) & (tseries["LONG"] >= 91.5))),
        # North East
        (tseries["LAT"] >= 16.5) & (tseries["LONG"] >= 89.5) |
        (tseries["LAT"] >= 22.5) & (tseries["LONG"] >= 83.5),
        # Western Himalayas
        (tseries["LAT"] >= 30.5),
        # Interior Peninsula
        ((tseries["LAT"] <= 21.5) 
        &
        ~((tseries["LAT"] <= 23.5) &
        (tseries["LONG"] <= 77.5) &
        (tseries["LAT"] + tseries["LONG"] < 95) &
        ~((tseries["LAT"] >= 13.5) & (tseries["LAT"] <= 17.5) & (tseries["LONG"] >= 76.5)) &
        ~((tseries["LAT"] >= 16.5) & (tseries["LAT"] <= 19.5) & (tseries["LONG"] >= 75.5)))
        &
        ~((tseries["LAT"] <= 22.5) &
        (tseries["LONG"] >= 77.5) &
        ~((tseries["LAT"] >= 9.5) & (tseries["LONG"] < 78.5)) &
        ~((tseries["LAT"] >= 14.5) & (tseries["LONG"] < 79.5)) &
        ~((tseries["LAT"] >= 17.5) & (tseries["LONG"] < 80.5)) &
        ~((tseries["LAT"] >= 18.5) & (tseries["LONG"] < 81.5)) &
        ~((tseries["LAT"] >= 19.5) & (tseries["LONG"] < 82.5)) &
        ~((tseries["LAT"] >= 20.5) & (tseries["LONG"] < 83.5)) &
        ~((tseries["LAT"] >= 21.5) & (tseries["LONG"] < 84.5)) &
        ~((tseries["LAT"] >= 22.5) & (tseries["LONG"] < 85.5)) &
        ~((tseries["LAT"] >= 20.5) & (tseries["LONG"] >= 91.5)))),
        # Northern Central
        ((tseries["LAT"] >= 21.5) & (tseries["LAT"] <= 30.5) & (tseries["LONG"] <= 83.5) &
        ~((tseries["LAT"] >= 21.5) & (tseries["LONG"] <= 73.5)) &
        ~((tseries["LAT"] >= 23.5) & (tseries["LONG"] <= 74.5)) &
        ~((tseries["LAT"] >= 24.5) & (tseries["LONG"] <= 75.5)) &
        ~((tseries["LAT"] >= 25.5) & (tseries["LONG"] <= 76.5)) &
        ~((tseries["LAT"] >= 26.5) & (tseries["LONG"] <= 77.5)) &
        ~((tseries["LAT"] >= 27.5) & (tseries["LONG"] <= 78.5)) &
        ~((tseries["LAT"] >= 28.5) & (tseries["LONG"] <= 79.5)) &
        ~((tseries["LAT"] >= 29.5) & (tseries["LONG"] <= 80.5))),
        # North West
        ((tseries["LAT"] >= 21.5) & (tseries["LAT"] <= 30.5) & (tseries["LONG"] <= 83.5) &
        ~(~((tseries["LAT"] >= 21.5) & (tseries["LONG"] <= 73.5)) &
        ~((tseries["LAT"] >= 23.5) & (tseries["LONG"] <= 74.5)) &
        ~((tseries["LAT"] >= 24.5) & (tseries["LONG"] <= 75.5)) &
        ~((tseries["LAT"] >= 25.5) & (tseries["LONG"] <= 76.5)) &
        ~((tseries["LAT"] >= 26.5) & (tseries["LONG"] <= 77.5)) &
        ~((tseries["LAT"] >= 27.5) & (tseries["LONG"] <= 78.5)) &
        ~((tseries["LAT"] >= 28.5) & (tseries["LONG"] <= 79.5)) &
        ~((tseries["LAT"] >= 29.5) & (tseries["LONG"] <= 80.5))))
    ]
    
    choices = ['WC', 'EC', 'NE', 'WH', 'IP', 'NC', 'NW']
    tseries['region'] = np.select(conditions, choices, default='None')
    
    print(tseries['region'].value_counts())
    return tseries
    

if __name__ == "__main__":
    
    # Read raw tmin and tmax data into csv format
    # tmin = read_panel_heat_data("tmin"), 
    tmax = read_panel_heat_data("tmax")
    print(tmax.head())
    
    tmax_marked = mark_regions(tmax)
    tmax_marked.to_csv(f"{DATA_DIR}/main_tmax_panel_withregions.csv", index=False)

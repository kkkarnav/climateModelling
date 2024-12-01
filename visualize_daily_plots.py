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


def create_daily_plots(tseries, tstring="Maximum"):
    date = datetime.datetime(1951, 1, 1)
    date_list = [(date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(26701)]
    
    tseries['geometry'] = tseries.progress_apply(lambda row: box(row['LONG'], row['LAT'], row['LONG'] + 1, row['LAT'] + 1), axis=1)
    tseries['TEMP'] = tseries['TEMP'].apply(lambda x: float(x) if x else None)
    
    for date in tqdm(date_list):
        
        try: 
            subset_tseries = tseries.loc[tseries["date"] == date, :]
            gdf = gpd.GeoDataFrame(subset_tseries, geometry='geometry')
            # gdf.to_file(f"./tmax_gdf.shp", driver='ESRI Shapefile')
           
            gdf.plot(column='TEMP', cmap='coolwarm', vmin=5, vmax=50, legend=True)
            plt.title(f'Daily {tstring} Temperatures for India on {date}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.savefig(f'{OUTPUT_DIR}/{date}.png')
            plt.clf()
            
        except Exception as e:
            print(f"Error in plotting {date}: {e}")
            continue
        
        
def create_daily_gif(dir):

    files = glob.glob(rf"{dir}/*.png")
    
    fig, ax = plt.subplots()
    im = ax.imshow(Image.open(files[0]), animated=True)
    
    ax.axis('off')
    
    def update(i):
        im.set_array(Image.open(files[10*i]))
        return im, 
    
    animation_fig = animation.FuncAnimation(fig, update, frames=round(len(files)/10)-10, interval=0.2, repeat_delay=100, blit=True)
    animation_fig.save("./images/animated_tmax_daily.gif")
    

if __name__ == "__main__":
    
    # Read raw tmin and tmax data into csv format
    # tmin = read_panel_heat_data("tmin")
    tmax = read_panel_heat_data("tmax")
    print(tmax.head())
    
    # create_daily_plots(tmax)
    create_daily_gif(OUTPUT_DIR)

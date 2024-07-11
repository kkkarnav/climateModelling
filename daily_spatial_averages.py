import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# Load your data
tmax_data = pd.read_csv("./per_grid_data/max_all.csv")

# Filter out extreme values
tmax_data = tmax_data[tmax_data['tmax'] <= 53]
tmax_data['time'] = pd.to_datetime(tmax_data['time'])
grid_size = 1
tmax_data['geometry'] = [box(lon, lat, lon + grid_size, lat + grid_size) for lon, lat in zip(tmax_data['lon'], tmax_data['lat'])]
gdf = gpd.GeoDataFrame(tmax_data, geometry='geometry')

india = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
india = india[india.name == "India"]

gdf['intersection_area'] = gdf.geometry.apply(lambda x: x.intersection(india.unary_union).area)
gdf['weight'] = gdf['intersection_area'] / gdf['geometry'].area

gdf['weighted_tmax'] = gdf['tmax'] * gdf['weight']
daily_weighted_avg = gdf.groupby(gdf['time'].dt.date).apply(
    lambda x: (x['weighted_tmax'].sum() / x['weight'].sum())
)
pd.DataFrame(daily_weighted_avg).to_csv("./daily_average_tmax.csv")


tmin_data = pd.read_csv("./per_grid_data/min_all.csv")
tmin_data = tmin_data[tmin_data['tmin'] <= 53]
tmin_data['time'] = pd.to_datetime(tmin_data['time'])
grid_size = 1
tmin_data['geometry'] = [box(lon, lat, lon + grid_size, lat + grid_size) for lon, lat in zip(tmin_data['lon'], tmin_data['lat'])]
gdf = gpd.GeoDataFrame(tmin_data, geometry='geometry')

india = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
india = india[india.name == "India"]

gdf['intersection_area'] = gdf.geometry.apply(lambda x: x.intersection(india.unary_union).area)
gdf['weight'] = gdf['intersection_area'] / gdf['geometry'].area

gdf['weighted_tmin'] = gdf['tmin'] * gdf['weight']
daily_weighted_avg = gdf.groupby(gdf['time'].dt.date).apply(
    lambda x: (x['weighted_tmin'].sum() / x['weight'].sum())
)
pd.DataFrame(daily_weighted_avg).to_csv("./daily_average_tmin.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import warnings
from datetime import datetime

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import roc_curve, roc_auc_score

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


def linear_regression(df):
    
    groups = df.groupby(['LAT', 'LONG'])
    predictions = {}

    for (lat, lon), group in groups:
        
        X = group[['year', 'day']].values
        y = group['TEMP'].values
        
        poly = PolynomialFeatures(degree=9)
        model = make_pipeline(poly, LinearRegression())
        model.fit(X, y)
        
        future_X = X[-1] + np.arange(1, 3661) / 365
        future_predictions = model.predict(future_X.reshape(-1, 1))
        
        # Store predictions
        predictions[(lat, lon)] = future_predictions
    
    average_predictions = np.zeros(3660)
    for key in predictions:
        average_predictions += predictions[key]
    average_predictions /= len(predictions)
    
    avg = df.groupby(["year", "day"])["TEMP"].mean()
    total = pd.concat([pd.Series(avg), pd.Series(average_predictions)]).reset_index(drop=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(total, label='Averages', color='dodgerblue')
    plt.legend(loc='lower left')
    plt.show()
    
    

if __name__ == "__main__":
    # Read raw tmin and tmax data into csv format
    tmax = pd.read_csv("./dataset/heat_main/main_tmax_panel.csv")
    
    tmax["year"] = pd.to_datetime(tmax["date"]).dt.year
    tmax["day"] = pd.to_datetime(tmax["date"]).dt.dayofyear
    
    tmax.drop(columns=["date"], inplace=True)
    tmax = tmax[tmax["TEMP"] != -99]
    
    print(tmax.head())
    
    linear_regression(tmax)

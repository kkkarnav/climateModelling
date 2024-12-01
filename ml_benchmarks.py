import xarray as xr
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

from tsai.all import *
from tsai.basics import *
from tsai.inference import load_learner


matplotlib.use('Agg')
tqdm.pandas()
warnings.filterwarnings("ignore")

OUTPUT_DIR = "./dataset/heat_main"


def read_panel_heat_data(variable):
    df = pd.read_csv(f"{OUTPUT_DIR}/main_{variable}_panel.csv").replace(-99, None)

    return df


def multi_regression(df):

    X, y, splits = get_regression_data('AppliancesEnergy', split_data=False)
    tfms = [None, TSRegression()]
    batch_tfms = TSStandardize(by_sample=True)
    reg = TSRegressor(X, y, splits=splits, path='models', arch="TSTPlus", tfms=tfms, batch_tfms=batch_tfms, metrics=rmse, verbose=True)
    print(reg)
    reg.fit_one_cycle(100, 3e-4)
    reg.export("reg.pkl")
    
    reg = load_learner("models/reg.pkl")
    raw_preds, target, preds = reg.get_X_preds(X[splits[1]], y[splits[1]])
    print(preds)


if __name__ == "__main__":
    
    # Read raw tmin and tmax data into csv format
    # tmin = read_panel_heat_data("tmin"), 
    tmax = read_panel_heat_data("tmax")
    
    tmax["year"] = pd.to_datetime(tmax["date"]).dt.year
    tmax["day"] = pd.to_datetime(tmax["date"]).dt.dayofyear
    tmax.drop(columns=["date"], inplace=True)
    tmax.sort_values(by=["year", "day", "LAT", "LONG"], inplace=True)
    
    tmax = tmax[tmax["TEMP"].notna()]
    print(tmax.head())
    
    # multi_regression(tmax)

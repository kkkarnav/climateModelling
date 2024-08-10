import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from datetime import datetime

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


def seasonal_decomposition(tseries, model):
    
    result = seasonal_decompose(tseries.iloc[:, -1], model=model, period=365)
    
    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    residual = result.resid.dropna()
    
    plt.figure(figsize=(10, 6))
    plt.suptitle(f"Seasonal Decomposition ({model})", fontsize=12, weight='bold')
    
    plt.subplot(4, 1, 1)
    plt.plot(tseries.iloc[:, -1], label='Original Data', color="cornflowerblue")
    plt.legend(loc='lower left')
    
    plt.subplot(4, 1, 2)
    plt.plot(trend, label=f'Trend ({model})', color='dodgerblue')
    plt.legend(loc='lower left')
    
    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label=f'Seasonal ({model})', color='mediumseagreen')
    plt.legend(loc='lower left')
    
    plt.subplot(4, 1, 4)
    plt.plot(residual, label=f'Residual ({model})', color='firebrick')
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f"./images/decompose_{model}.png", dpi=600, bbox_inches="tight")
    plt.clf()
        

def stl_decomposition(tseries, robust):
    
    result = STL(tseries.iloc[:, -1], period=365, seasonal=11, robust=robust).fit()
    
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid
    
    plt.figure(figsize=(10, 6))
    plt.suptitle(f"STL Decomposition, robust={robust}", fontsize=12, weight='bold')
    
    plt.subplot(4, 1, 1)
    plt.plot(tseries.iloc[:, -1], label='Original Data', color="cornflowerblue")
    plt.legend(loc='lower left')
    
    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend (STL)', color='dodgerblue')
    plt.legend(loc='lower left')
    
    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal (STL)', color='mediumseagreen')
    plt.legend(loc='lower left')
    
    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residual (STL)', color='firebrick')
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f"./images/decompose_stl_robust_{robust}.png", dpi=600, bbox_inches="tight")
    plt.clf()
    
    show_weights = False
    if robust and show_weights == True:
        plt.figure(figsize=(10, 6))
        plt.plot(result.weights, marker="o", linestyle="none")
        plt.savefig(f"./images/decompose_stl_robust_weights.png", dpi=600, bbox_inches="tight")
        plt.clf()
    

def stl_forecast(tseries):
    
    stlf = STLForecast(tseries.iloc[:-1825, -1], ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), period=365)
    stlf_res = stlf.fit()
    forecast = stlf_res.forecast(3650)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tseries.iloc[:, -1], alpha=0.5, color="cornflowerblue")
    plt.plot(forecast, alpha=0.5, color="firebrick")
    plt.title("STL Forecast", weight='bold')
    plt.savefig(f"./images/forecast_stl_full.png", dpi=600, bbox_inches="tight")
    plt.clf()
    
    plt.figure(figsize=(10, 6))
    plt.plot(tseries.iloc[-3650:, -1], alpha=0.5, color="cornflowerblue")
    plt.plot(forecast, alpha=0.5, color="firebrick")
    plt.title("STL Forecast", weight='bold')
    plt.savefig(f"./images/forecast_stl.png", dpi=600, bbox_inches="tight")
    plt.clf()
    
    result = STL(forecast, period=365, seasonal=11).fit()
    
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid
    
    plt.figure(figsize=(10, 6))
    plt.suptitle(f"STL Forecast", fontsize=12, weight='bold')
    
    plt.subplot(4, 1, 1)
    plt.plot(forecast, label='Actual Forecast', color="cornflowerblue")
    plt.legend(loc='lower left')
    
    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend (STL)', color='dodgerblue')
    plt.legend(loc='lower left')
    
    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal (STL)', color='mediumseagreen')
    plt.legend(loc='lower left')
    
    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residual (STL)', color='firebrick')
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f"./images/decompose_stl_forecast.png", dpi=600, bbox_inches="tight")
    plt.clf()
    
    print(stlf_res.summary())


def test_adf(tseries):
    
    print("\nDickey-Fuller Test:\n")
    dftest = adfuller(tseries, autolag="AIC")
    
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    
    print(dfoutput)


def test_kpss(tseries):
    
    print("\nKPSS Test:\n")
    kpsstest = kpss(tseries, regression="c", nlags="auto")
    
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    
    print(kpss_output)


def autocorrelation(tseries, lags):
    plt.figure(figsize=(10, 6))
    plt.title("Autocorrelation of daily temperature with previous 1-30 days readings")
    plot_acf(tseries, lags=lags, color="steelblue")
    
    plt.tight_layout()
    plt.savefig(f"./images/autocorrelation_{str(lags)}.png", dpi=600, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    # Read raw tmin and tmax data into csv format
    tmax = read_heat_data("tmax")
    tmax.index = pd.to_datetime(tmax[("lat", "lon")])

    # This converts into a single series by taking the average of all columns
    # Find a better way to do this
    tmax[("avg", "avg")] = tmax.iloc[:, 1:].mean(axis=1)
    print(tmax.head())
    
    # seasonal_decomposition(tmax, "additive")
    # seasonal_decomposition(tmax, "multiplicative")
    # stl_decomposition(tmax, True)
    # stl_decomposition(tmax, False)
    
    stl_forecast(tmax)
    
    # Dickey Fuller and KPSS indicate that the series is stationary
    # test_adf(tmax[("avg", "avg")])
    # test_kpss(tmax[("avg", "avg")])
    
    # Data is strongly autocorrelated with the preceding readings (as expected)
    # autocorrelation(tmax.iloc[:, -1], 30)
    # autocorrelation(tmax.iloc[:, -1], 365)

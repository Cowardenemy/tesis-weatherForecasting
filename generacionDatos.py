import os
import pandas as pd
import numpy as np
import math
import scipy
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D
from numpy import hstack
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.colors as mcolors
from sktime.distances import (
    dtw_distance,
    ddtw_distance,
    wdtw_distance,
    msm_distance,
    erp_distance,
    lcss_distance,
    twe_distance,
    wddtw_distance,
    edr_distance
)
def foxmethod(targetWindow, windows):
    pearsonCorrelation = np.array(
        ([np.corrcoef(windows[currentWindow, :, currentComponent], targetWindow[:, currentComponent])[0][1]
          for currentWindow in range(len(windows)) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                        componentsLen)
    euclideanDistance = np.array(
        ([np.linalg.norm(targetWindow[:, currentComponent] - windows[currentWindow, :, currentComponent])
          for currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                         componentsLen)
    normalizedEuclideanDistance = euclideanDistance / np.amax(euclideanDistance, axis=0)
    normalizedCorrelation = (.5 + (pearsonCorrelation - 2 * normalizedEuclideanDistance + 1) / 4)
    correlationPerWindow = np.sum(((normalizedCorrelation + punishedSumFactor) ** 2), axis=1)
    correlationPerWindow = scaler.fit_transform(correlationPerWindow.reshape(-1, 1)).reshape(1, -1)[0]
    return correlationPerWindow

def split_sequences(sequences, n_steps):
    inputnn, target = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix + 1 > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix], sequences[end_ix, (1,4,7)]
        inputnn.append(seq_x)
        target.append(seq_y)
    return array(inputnn), array(target)

laps = 49

nonOutputColumns=[0,2,3,5,6,8]
punishedSumFactor = .5
df_temp = pd.read_csv(os.path.join(os.path.dirname(__file__), "weatherdata.csv"), parse_dates= True, index_col= 1)
df_temp.head()
dataset = df_temp.filter(['HUM_MIN','HUM_AVG','HUM_MAX','PRES_MIN','PRES_AVG','PRES_MAX','TEMP_MIN','TEMP_AVG','TEMP_MAX']).values
dataset = np.array(dataset)

for step_days in range(10,laps+2,1):
    inputnn, target = split_sequences(dataset, step_days)
    windows = np.delete(inputnn, nonOutputColumns, 2)
    targetWindow, windows = windows[-1], windows[:-1]
    windowsLen = len(windows)
    componentsLen = windows.shape[2]
    windowLen = windows.shape[1]

    print("Longitud de ventana actual: ", step_days)
    scaler = MinMaxScaler()
    print("CCI")
    correlationPerWindow = foxmethod(targetWindow, windows)
    print("DTW")
    DTW = np.array(([dtw_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                     currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                componentsLen)
    print("DDTW")
    DDTW = np.array(([ddtw_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                      currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                 componentsLen)
    print("MSM")
    MSM = np.array(([msm_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                     currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                componentsLen)
    print("ERP")
    ERP = np.array(([erp_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                     currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                componentsLen)
    print("LCSS")
    LCSS = np.array(([lcss_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                      currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                 componentsLen)
    print("TWE")
    TWE = np.array(([twe_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                     currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                componentsLen)
    print("EDR")
    EDR = np.array(([edr_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                     currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                componentsLen)
    allMethods = np.array([DTW, DDTW, MSM, ERP, LCSS, TWE, EDR])
    np.save('allMethods' + str(step_days) + 'Win.npy', allMethods)
    DTW, DDTW, MSM, ERP, LCSS, TWE, EDR = None, None, None, None, None, None, None
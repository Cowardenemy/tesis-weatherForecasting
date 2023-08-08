import logging
import threading
import time
import os
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
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
def foxmethod(targetWindow, windows,componentsLen):
    windowsLen = len(windows)
    componentsLen = windows.shape[2]
    windowLen = windows.shape[1]
    punishedSumFactor = .5
    scaler = MinMaxScaler()
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


def thread_function(name, targetLap):
    logging.info("Thread %s: starting", name)
    laps = 49
    nonOutputColumns = [0, 2, 3, 5, 6, 8]
    punishedSumFactor = .5
    df_temp = pd.read_csv("weatherdata.csv", parse_dates=True, index_col=1)
    df_temp.head()
    dataset = df_temp.filter(
        ['HUM_MIN', 'HUM_AVG', 'HUM_MAX', 'PRES_MIN', 'PRES_AVG', 'PRES_MAX', 'TEMP_MIN', 'TEMP_AVG',
         'TEMP_MAX']).values
    dataset = np.array(dataset)

    for step_days in range(targetLap, laps + 2, 1):
        inputnn, target = split_sequences(dataset, step_days)
        windows = np.delete(inputnn, nonOutputColumns, 2)
        targetWindow, windows = windows[-1], windows[:-1]
        windowsLen = len(windows)
        componentsLen = windows.shape[2]
        windowLen = windows.shape[1]

        print("Longitud de ventana actual: ", step_days)
        scaler = MinMaxScaler()
        print("CCI " + str(step_days))
        correlationPerWindow = foxmethod(targetWindow, windows, componentsLen)
        print("DTW " + str(step_days))
        DTW = np.array(
            ([dtw_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
              currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                         componentsLen)
        print("DDTW " + str(step_days))
        DDTW = np.array(
            ([ddtw_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
              currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                         componentsLen)
        print("MSM " + str(step_days))
        MSM = np.array(
            ([msm_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
              currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                         componentsLen)
        print("ERP " + str(step_days))
        ERP = np.array(
            ([erp_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
              currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                         componentsLen)
        print("LCSS " + str(step_days))
        LCSS = np.array(
            ([lcss_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
              currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                         componentsLen)
        print("TWE " + str(step_days))
        TWE = np.array(
            ([twe_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
              currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                         componentsLen)
        print("EDR " + str(step_days))
        EDR = np.array(
            ([edr_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
              currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                         componentsLen)
        allMethods = np.array([DTW, DDTW, MSM, ERP, LCSS, TWE, EDR])
        np.save('allMethods' + str(step_days) + 'Win.npy', allMethods)
        DTW, DDTW, MSM, ERP, LCSS, TWE, EDR = None, None, None, None, None, None, None
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread")
    x = threading.Thread(target=thread_function, args=(1,32))
    y = threading.Thread(target=thread_function, args=(2,33))
    z = threading.Thread(target=thread_function, args=(3,34))
    logging.info("Main    : before running thread")
    x.start()
    y.start()
    z.start()
    # x.join()
    logging.info("Main    : all done")
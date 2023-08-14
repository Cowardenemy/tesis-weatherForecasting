import os
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sktime.distances import (
    dtw_distance,
    ddtw_distance,
    msm_distance,
    erp_distance,
    lcss_distance,
    twe_distance,
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

laps = 28

nonOutputColumns=[0,2,3,5,6,8]
punishedSumFactor = .5
df_temp = pd.read_csv(os.path.join(os.path.dirname(__file__), "weatherdata.csv"), parse_dates= True, index_col= 1)
df_temp.head()
dataset = df_temp.filter(['HUM_MIN','HUM_AVG','HUM_MAX','PRES_MIN','PRES_AVG','PRES_MAX','TEMP_MIN','TEMP_AVG','TEMP_MAX']).values
dataset = np.array(dataset)


for step_days in range(27,laps+2,1):
    inputnn, target = split_sequences(dataset, step_days)
    windows = np.delete(inputnn, nonOutputColumns, 2)
    targetWindow, windows = windows[-1], windows[:-1]
    windowsLen = len(windows)
    componentsLen = windows.shape[2]
    windowLen = windows.shape[1]

    print("Longitud de ventana actual: ", step_days)
    scaler = MinMaxScaler()
    for i in range(8):
        match i:
            case 1:
                print("CCI")
                metodo = foxmethod(targetWindow, windows)
                np.save('CCI' + str(step_days) + 'Win.npy', metodo)
                print("supuestamente save")
            case 2:
                print("DTW")
                metodo = np.array(
                    ([dtw_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                      currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1, componentsLen)
                np.save('DTW' + str(step_days) + 'Win.npy', metodo)
            case 3:
                print("DDTW")
                metodo = np.array(
                    ([ddtw_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                      currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                 componentsLen)
                np.save('DDTW' + str(step_days) + 'Win.npy', metodo)
            case 4:
                print("MSM")
                metodo = np.array(
                    ([msm_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                      currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                 componentsLen)
                np.save('MSM' + str(step_days) + 'Win.npy', metodo)
            case 5:
                print("ERP")
                metodo = np.array(
                    ([erp_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                      currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                 componentsLen)
                np.save('ERP' + str(step_days) + 'Win.npy', metodo)
            case 6:
                print("LCSS")
                metodo = np.array(
                    ([lcss_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                      currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                 componentsLen)
                np.save('LCSS' + str(step_days) + 'Win.npy', metodo)
            case 7:
                print("TWE")
                metodo = np.array(
                    ([twe_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                      currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                 componentsLen)
                np.save('TWE' + str(step_days) + 'Win.npy', metodo)
            case 8:
                print("EDR")
                metodo = np.array(
                    ([edr_distance(targetWindow[:, currentComponent], windows[currentWindow, :, currentComponent]) for
                      currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                                 componentsLen)
                np.save('EDR' + str(step_days) + 'Win.npy', metodo)







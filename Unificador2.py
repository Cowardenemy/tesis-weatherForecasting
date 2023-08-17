import os
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler

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

nonOutputColumns=[0,2,3,5,6,8]
punishedSumFactor = .5
df_temp = pd.read_csv("weatherdata.csv", parse_dates= True, index_col= 1)
df_temp.head()
dataset = df_temp.filter(['HUM_MIN','HUM_AVG','HUM_MAX','PRES_MIN','PRES_AVG','PRES_MAX','TEMP_MIN','TEMP_AVG','TEMP_MAX']).values
dataset = np.array(dataset)
scaler = MinMaxScaler()
for step_days in range(3,36,1):
    inputnn, target = split_sequences(dataset, step_days)
    windows = np.delete(inputnn, nonOutputColumns, 2)
    targetWindow, windows = windows[-1], windows[:-1]
    windowsLen = len(windows)
    componentsLen = windows.shape[2]
    windowLen = windows.shape[1]
    try:
        CCI = foxmethod(targetWindow, windows)
        CCI[np.isnan(CCI)] = 0
        print(len(CCI))
        todos = np.load("allMethods"+str(step_days)+"Win.npy")
        allMethodsNorm = []
        for i in range(len(todos)):
            allMethodsNorm.append(scaler.fit_transform(np.sum(scaler.fit_transform(todos[i]), axis=1).reshape(-1, 1)).reshape(1, -1)[0] * -1 + 1)
        allMethodsNorm = np.array(allMethodsNorm)
        TCCI =[]
        TCCI.append(CCI)
        metodos = np.concatenate((allMethodsNorm,np.array(TCCI)), axis=0)
        np.save('Metodos/trueAllMethods' + str(step_days) + 'Win.npy', metodos)
        print("Guardado")
    except:
        print("Algo falló")
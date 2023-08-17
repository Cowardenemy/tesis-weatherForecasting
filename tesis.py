import matplotlib.pyplot
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
def combinedOption(op,bestList,kMax,indexMethods):
    color=[".-b",".-g",".-m"]
    color2=["db","dg","dm"]
    plt.figure(figsize=(12,8))
    newCase = np.zeros((windowLen,componentsLen))
    newWeigthedCase = np.zeros((windowLen,componentsLen))
    partialCase = []
    try:
        if op==1:
            newCase = np.sum(windows[list(dict(allMbestSorted[indexMethods][:kMax]).keys())], axis=0)/kMax
            simSum = 0
            for i in range(kMax):
                partialCase.append(windows[allMbestSorted[indexMethods][i][0]]*allMbestSorted[indexMethods][i][1])
                simSum+=allMbestSorted[indexMethods][i][1]
            newWeigthedCase=np.sum(partialCase,axis=0)/simSum
        elif op==2:
            newCase = np.max(windows[list(dict(bestList).keys())], axis=0)
        elif op==3:
            newCase = np.min(windows[list(dict(bestList).keys())], axis=0)
        elif op==4:
            newCase = np.median(windows[list(dict(bestList).keys())], axis=0)
    except:
        print("Unavailable option")
    for f in range(componentsLen):
        plt.subplot(componentsLen,1,f+1)
        plt.title(titleColumns[f])
        plt.plot(cont,targetWindow[:,f], '.-k', label = "Target")
        plt.plot(cont,newCase[:,f], color[f] ,label= "Data")
        plt.plot(windowLen+1,actualPrediction[f], 'dk', label = "Prediction")
        plt.plot(windowLen+1,newCase[windowLen-1][f], color2[f], label = "Next day")
        plt.grid()
        plt.xticks(range(1,windowLen+2,1))
        plt.yticks(lims[f])
    plt.tight_layout()
    plt.savefig("Resultados/Figures/CombinedOption "+methodNames[indexMethods]+".png")
    matplotlib.pyplot.close("all")
    return newCase,newWeigthedCase
def calculateDistSimilarity(combinedSlice):
    allMdists = []
    for k in range(len(allMethodsNorm)):
        allMdist = []
        for i in range(componentsLen):
            allMdist.append(math.dist(targetWindow[:,i], combinedSlice[k][:,i]))
        allMdists.append(allMdist)
    allMaverageSimilarity = []
    for i in range(len(allMdists)):
        allMaverageSimilarity.append(allMdists[i])
    #allMaverageSimilarity.append([math.dist(targetWindow[:,i], np.array(np.sum(windows[list(dict(bestSorted[:kMax]).keys())], axis=0)/len(bestSorted[:kMax]))[:,i]) for i in range(componentsLen)])
    allMaverageSimilarity = np.array(allMaverageSimilarity)
    for i in range(componentsLen):
        allMaverageSimilarity[:,i]=allMaverageSimilarity[:,i] / (componentsLims[i][0]-componentsLims[i][1])

    return allMaverageSimilarity

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





df_temp = pd.read_csv("C:/Users/cowar/OneDrive/Documents/GitHub/tesis-weatherForecasting/weatherdata.csv", parse_dates= True, index_col= 1)
df_temp.head()

dataset = df_temp.filter(['HUM_MIN','HUM_AVG','HUM_MAX','PRES_MIN','PRES_AVG','PRES_MAX','TEMP_MIN','TEMP_AVG','TEMP_MAX']).values
dataset = np.array(dataset)
methodNames = ["Dynamic Time Warping","Derivate Dynamic Time Warping",
               "Move Split Merge","Edit Distance For Real Penalty","Longest Common Subsequence","Time Warp Edit",
              "Edit Distance for Real Sequences","Combined Correlation Index"]

rmseTrain,rmseTest,uTest,uTestDTW  = [],[],[],[]
meanBestMAE,meanBestMAEDTW,meanWorstMAE,meanWorstMAEDTW = [],[],[],[]
scaler = MinMaxScaler()

for step_days in range(3,36,1):
   cont = np.arange(1, step_days + 1)
   print("Longitud de ventana actual: ",step_days)
#División de dataset en ventanas de n longitud
   inputnn, target = split_sequences(dataset, step_days)
   input_train, input_test, target_train, target_test = train_test_split(inputnn, target, test_size = 0.30, random_state=4,shuffle=True)
   model = Sequential()
#Creación del modelo con la longitud actual de ventanas
   model.add(LSTM(100, activation = 'relu', return_sequences = True, input_shape = (step_days ,input_train.shape[2])))
   model.add(LSTM(32, activation = 'relu', return_sequences=False))
   model.add(Dropout(0.2))
   model.add(Dense(3))
   model.compile(optimizer = 'adam', loss = 'mse')
   history = model.fit(input_train, target_train, validation_data = (input_test, target_test),batch_size = 16, epochs = 100)
#RMSE de train  y test
   prediction_train = model.predict(input_train)
   RMSE = math.sqrt(np.square(np.subtract(prediction_train, target_train)).mean())
   rmseTrain.append(RMSE)


   prediction = model.predict(input_test)
   RMSE = math.sqrt(np.square(np.subtract(prediction, target_test)).mean())
   rmseTest.append(RMSE)
   input_train, input_test, target_train, target_test = None,None,None,None
#Declaración de variables
   nonOutputColumns=[0,2,3,5,6,8]
   windows = np.delete(inputnn, nonOutputColumns, 2)
   targetWindow, windows = windows[-1], windows[:-1]
   windowsLen = len(windows)
   componentsLen = windows.shape[2]
   windowLen = windows.shape[1]
   actualPrediction = prediction[-1]
   titleColumns = ["Humidity", "Vapor Pressure" ,"Temperature"]
   titleIndexes = ["Window Index {0}".format(index) for index in range(windowsLen)]
   smoothnessFactor = .03
   punishedSumFactor = .5
   finalWindowNumber = 30
   #0 number of results, 1 average, 2 Max values, 3 min values, 4 median
   explicationMethodResult = 1
   componentsLims = [(np.amax(windows[:, :, i]), np.amin(windows[:, :, i])) for i in range(componentsLen)]
#Límites de cada componente
   maxComp, minComp, lims = [], [], []
   for i in range(componentsLen):
       maxComp.append(int(max(max(a) for a in windows[:, :, i])))
       minComp.append(int(min(min(a) for a in windows[:, :, i])))
       lims.append(range(minComp[i], maxComp[i], int((maxComp[i] - minComp[i]) / 8)))
#Predicción normalizada
   normalizedActualPrediction = (actualPrediction - minComp) / maxComp
#Evaluación de similitud
   allMethodsNorm = np.load("Metodos/trueAllMethods"+str(step_days)+"Win.npy")
#Reducción de ruido
   smoothedMethods = []
   for i in range(len(allMethodsNorm)):
       smoothedMethods.append(lowess(allMethodsNorm[i], np.arange(len(allMethodsNorm[i])), smoothnessFactor)[:, 1])
   smoothedMethods = np.array(smoothedMethods)

#Localización de picos y valles
   allMvalleyIndex, allMpeakIndex = [], []
   for i in range(len(smoothedMethods)):
       allMvalleyIndex.append(signal.argrelextrema(smoothedMethods[i], np.less)[0])
       allMpeakIndex.append(signal.argrelextrema(smoothedMethods[i], np.greater)[0])

   allMconcaveSegments, allMconvexSegments = [], []
   for i in range(len(allMethodsNorm)):
       allMconcaveSegments.append(
           np.split(np.transpose(np.array((np.arange(windowsLen), allMethodsNorm[i]))), allMvalleyIndex[i]))
       allMconvexSegments.append(
           np.split(np.transpose(np.array((np.arange(windowsLen), allMethodsNorm[i]))), allMpeakIndex[i]))
   bestWindowsIndex, worstWindowsIndex, allMbestWindowsIndex, allMworstWindowsIndex = list(), list(), list(), list()
   for i in range(len(allMconcaveSegments)):
       currentMethodSplitConc, currentMethodSplitConv = [], []
       for split in allMconcaveSegments[i]:
           currentMethodSplitConc.append(int(split[np.where(split == max(split[:, 1]))[0][0], 0]))
       for split in allMconvexSegments[i]:
           currentMethodSplitConv.append(int(split[np.where(split == min(split[:, 1]))[0][0], 0]))
       allMbestWindowsIndex.append(currentMethodSplitConc)
       allMworstWindowsIndex.append(currentMethodSplitConv)
#Detección del mejor y peor valor para cada segmento
   for i in range(len(allMconcaveSegments)):
       currentMethodSplitConc, currentMethodSplitConv = [], []
       for split in allMconcaveSegments[i]:
           currentMethodSplitConc.append(int(split[np.where(split == np.nanmax(split[:, 1]))[0][0], 0]))
       for split in allMconvexSegments[i]:
           currentMethodSplitConv.append(int(split[np.where(split == np.nanmin(split[:, 1]))[0][0], 0]))
       allMbestWindowsIndex.append(currentMethodSplitConc)
       allMworstWindowsIndex.append(currentMethodSplitConv)
   allMconcaveSegments, allMconvexSegments = None,None
   allMbestDic, allMworstDic, allMbestSorted, allMworstSorted = [], [], [], []
   for i in range(len(allMethodsNorm)):
       allMbestDic.append({index: allMethodsNorm[i][index] for index in allMbestWindowsIndex[i]})
       allMworstDic.append({index: allMethodsNorm[i][index] for index in allMworstWindowsIndex[i]})
   for i in range(len(allMethodsNorm)):
       allMbestSorted.append(sorted(allMbestDic[i].items(), reverse=True, key=lambda x: x[1]))
       allMworstSorted.append(sorted(allMworstDic[i].items(), key=lambda x: x[1]))
#Comparación con el siguiente día de la ventana target con el de las demás ventanas
   allMbestMAE, allMworstMAE = [], []
   for k in range(len(allMethodsNorm)):
       subBestMAE, subWorstMAE = [], []
       for i in range(len(allMbestSorted[k])):
           rawBestMAE = rawWorstMAE = 0
           for f in range(componentsLen):
               rawBestMAE += (((windows[allMbestSorted[k][i][0]][windowLen - 1][f] - minComp[f]) / maxComp[f]) -
                              normalizedActualPrediction[f])
               rawWorstMAE += ((windows[allMworstSorted[k][i][0]][windowLen - 1][f] - minComp[f]) / maxComp[f] -
                               normalizedActualPrediction[f])
           subBestMAE.append(rawBestMAE / componentsLen)
           subWorstMAE.append(rawWorstMAE / componentsLen)
       allMbestMAE.append(subBestMAE)
       allMworstMAE.append(subWorstMAE)
   import json
   with open("MAE_B_"+str(step_days), "w") as fp:
       json.dump(allMbestMAE, fp)
   with open("MAE_W_" + str(step_days), "w") as fp:
       json.dump(allMworstMAE, fp)
   #with open("test", "rb") as fp:   # Unpickling
       #b = pickle.load(fp)
   zValues = []
   for i in range(len(allMethodsNorm)):
       U1,pValue = scipy.stats.mannwhitneyu(allMbestMAE[i], allMworstMAE[i], method="exact")
       nx, ny = len(allMbestMAE[i]), len(allMworstMAE[i])
       z = (U1 - (nx*ny /2)) / np.sqrt(nx*ny * (nx + ny + 1)/ 12)
       zValues.append(z)
   np.save("zValues"+str(step_days)+"Win.npy",np.array(zValues))
   d = {'Method': methodNames}
   dw = {'Method': methodNames}
   columns = [1, 2, 3, 5, 7, 10, 15]
   for kMax in [1, 2, 3, 5, 7, 10, 15]:
       # print(kMax)
       newCases = []
       newWeightedCases = []
       for i in range(len(allMethodsNorm)):
           a, b = combinedOption(explicationMethodResult, allMbestSorted[i], kMax, i)
           newCases.append(a)
           newWeightedCases.append(b)
       allMaverageMAE = calculateDistSimilarity(newCases)
       allMWaverageMAE = calculateDistSimilarity(newWeightedCases)
       # allMWaverageMAE = (allMWaverageMAE-np.min(allMWaverageMAE))/(np.max(allMWaverageMAE)-np.min(allMWaverageMAE))
       # print(allMaverageMAE)
       # allMWaverageMAE = calculateWeigthedDistMAE(newWeightedCases)
       result = [math.sqrt(sum([k ** 2 for k in i])) for i in allMaverageMAE]
       resultW = [math.sqrt(sum([k ** 2 for k in i])) for i in allMWaverageMAE]
       # print(result)
       d[kMax] = result
       dw[kMax] = resultW

   dfw = pd.DataFrame.from_dict(data=dw)
   df = pd.DataFrame.from_dict(data=d)
   df.to_csv('C:/Users/cowar/OneDrive/Documents/GitHub/tesis-weatherForecasting/Resultados/Ponderados/ResultadosSinPonderar'+ str(step_days) + '.csv')
   dfw.to_csv('C:/Users/cowar/OneDrive/Documents/GitHub/tesis-weatherForecasting/Resultados/Ponderados/ResultadosPonderados'+ str(step_days) + '.csv')
#df = pd.DataFrame(data=np.transpose((rmseTrain,rmseTest,uTest,uTestDTW)),columns = ("RMSE Train","RMSE Test","Z Value", "Z value DTW"), index = np.arange(2,laps+2))
#df.to_csv('C:/Users/cowar/- tss/Resultados/Resultados.csv')
#print(df)
df_mae = pd.DataFrame(data=np.transpose((meanBestMAE,meanBestMAEDTW,meanWorstMAE,meanWorstMAEDTW)),columns = ("Best MAE","Best MAE DTW","Worst MAE", "Worst MAE DTW"), index = np.arange(2,laps+2))
df_mae.to_csv('C:/Users/cowar/OneDrive/Documents/GitHub/tesis-weatherForecasting/Resultados/MAE/ResultadosMAE.csv')
print(df_mae)

plt.figure(figsize=(15,5))
plt.plot(cont,rmseTrain, color = 'c', label='RMSE Train')
plt.plot(cont,rmseTest, color = 'b', label='RMSE Test')
plt.plot(cont,uTest, color = 'g', label='Z Value')
plt.plot(cont,uTestDTW, color = 'k', label='Z Value DTW')
plt.xticks(range(1,laps+1,1))
plt.legend()
plt.grid()
plt.savefig('C:/Users/cowar/OneDrive/Documents/GitHub/tesis-weatherForecasting/Resultados/Figures/Resultados_grafica.png')


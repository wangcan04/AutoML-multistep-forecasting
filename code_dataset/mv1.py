import numpy as np
import os
import autosklearn.regression
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')
import sys
import sklearn
from tqdm import tqdm
#list of 20 datasets
data=np.loadtxt('timeseries(not real)/Autoregressive with noise/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(not real)/Correlated noise/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(not real)/Diffusionless Lorenz Attractor/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(not real)/Driven pendulum with dissipation/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(not real)/Driven van der Pol oscillator/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(not real)/Duffing two-well oscillator/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(not real)/Duffing-van der Pol Oscillator/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(not real)/Moving average process/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(not real)/Nonstationary autoregressive/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(not real)/Random walk/comp-engine-export-datapoints.txt',delimiter=',')

#data=np.loadtxt('timeseries(real)/Crude oil prices/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(real)/ECG/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(real)/Exchange rate/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(real)/Gas prices/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(real)/human speech/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(real)/Macroeconomics/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(real)/Micoeconomics/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(real)/music/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(real)/Tropical forest soundscape(animal sound)/comp-engine-export-datapoints.txt',delimiter=',')
#data=np.loadtxt('timeseries(real)/Zooplankton growth/comp-engine-export-datapoints.txt',delimiter=',')




# Forecasting window size
window_size = 1
# The data points skip window. The first 200 data points will be skipped
# to make sure every point that being forecasted has enough input data. 
windowsize = 200
y = data
y = y[windowsize:]
X = []
for i in range(len(data)-windowsize):
    x = data[i:i+windowsize]
    X.append(x)
X = np.array(X)
# train, test split based on the temporal order
X_trainlist = X[:int(len(X)*0.67),-window_size:]
y_trainlist = y[:int(len(X)*0.67)]
X_testlist = X[int(len(X)*0.67):,-window_size:]
y_testlist = y[int(len(X)*0.67):]

predictions =  np.mean(X_trainlist, axis=1)



import math
score = sklearn.metrics.mean_squared_error(y_trainlist, predictions)
MAE = sklearn.metrics.mean_absolute_error(y_trainlist, predictions)
RMSEscore = math.sqrt(score)
R2 = sklearn.metrics.r2_score(y_trainlist, predictions)
print("ERRORS TRAIN SET:")
print('RMSE: ', str(RMSEscore))
print('MAE: ', str(MAE))
print('R2:', str(R2))


predictions =  np.mean(X_testlist, axis=1)

score = sklearn.metrics.mean_squared_error(y_testlist, predictions)
MAE = sklearn.metrics.mean_absolute_error(y_testlist, predictions)
RMSEscore = math.sqrt(score)
R2 = sklearn.metrics.r2_score(y_testlist, predictions)
print("ERRORS TEST SET:")
print ('RMSE: ', str(RMSEscore))
print ('MAE: ', str(MAE))
print('R2:', str(R2))




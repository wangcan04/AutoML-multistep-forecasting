#!/usr/bin/env python

from time import time
import numpy as np
import os
import datetime
import warnings
warnings.filterwarnings('ignore')
import sys
import sklearn
from tqdm import tqdm
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import math
import autokeras as ak
if __name__ == "__main__":
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
 window_size = 10 

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
 # max_trials indicates how many models will be evaluated
 reg = ak.StructuredDataRegressor(
    overwrite=True, max_trials=10)
 reg.fit(
    X_trainlist,
    y_trainlist,
 )
 predicted_y = reg.predict(X_testlist)


 predicted_y = reg.predict(X_trainlist)
 predictions = predicted_y
 score = sklearn.metrics.mean_squared_error(y_trainlist, predictions)
 MAE = sklearn.metrics.mean_absolute_error(y_trainlist, predictions)
 RMSEscore = math.sqrt(score)
 R2 = sklearn.metrics.r2_score(y_trainlist, predictions)
 print("ERRORS TRAIN SET:")
 print('RMSE: ', str(RMSEscore))
 print('MAE: ', str(MAE))
 print('R2:', str(R2))

 predicted_y = reg.predict(X_testlist)
 predictions = predicted_y
 score = sklearn.metrics.mean_squared_error(y_testlist, predictions)
 MAE = sklearn.metrics.mean_absolute_error(y_testlist, predictions)
 RMSEscore = math.sqrt(score)
 R2 = sklearn.metrics.r2_score(y_testlist, predictions)
 print("ERRORS TEST SET:")
 print ('RMSE: ', str(RMSEscore))
 print ('MAE: ', str(MAE))
 print('R2:', str(R2))
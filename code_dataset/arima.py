#!/usr/bin/env python


from time import time
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
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
from sklearn.datasets import fetch_california_housing
import math
import pmdarima as pm
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
 # dataset 
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
 train, test = model_selection.train_test_split(data, train_size=int(len(data)*0.67))

 modl = pm.auto_arima(train, error_action='ignore', trace=True,
                      stepwise=False)
 # Create predictions for the future, evaluate on test
 predictions, conf_int = modl.predict(n_periods=train.shape[0], return_conf_int=True)
 # Print the error:
 score = sklearn.metrics.mean_squared_error(train, predictions)
 MAE = sklearn.metrics.mean_absolute_error(train, predictions)
 RMSEscore = math.sqrt(score)
 R2 = sklearn.metrics.r2_score(train, predictions)
 print("ERRORS TRAIN SET:")
 print('RMSE: ', str(RMSEscore))
 print('MAE: ', str(MAE))
 print('R2:', str(R2))
 predictions, conf_int = modl.predict(n_periods=test.shape[0], return_conf_int=True)
 score = sklearn.metrics.mean_squared_error(test, predictions)
 MAE = sklearn.metrics.mean_absolute_error(test, predictions)
 RMSEscore = math.sqrt(score)
 R2 = sklearn.metrics.r2_score(test, predictions)
 print("ERRORS TEST SET:")
 print ('RMSE: ', str(RMSEscore))
 print ('MAE: ', str(MAE))
 print('R2:', str(R2))
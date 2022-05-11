import h2o
from h2o.automl import H2OAutoML

import numpy as np
import warnings
import os
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')
import sys
import sklearn
from tqdm import tqdm
import sklearn.model_selection
import sklearn.metrics
import h2o
warnings.filterwarnings(action='ignore', message='Setting attributes')
h2o.init()

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
 window_size = 200 

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

 train = np.c_[X_trainlist,y_trainlist]
test = np.c_[X_testlist,y_testlist]

train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)

aml = H2OAutoML(max_runtime_secs = 2*3600, seed = 1,max_runtime_secs_per_model=60*20,\
include_algos = ['XGBoost'], \
sort_metric='rmse')
aml.train(y = window_size,training_frame = train)
predictions = aml.predict(test)
predictions = np.array(predictions)
y_testlist = np.array(y_testlist)
print (aml.leader.model_performance(test))
#!/usr/bin/env python
import numpy as np
import os
import autosklearn.regression
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

from smac.facade.roar_facade import ROAR
from smac.scenario.scenario import Scenario

import autosklearn.classification


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



 i=int(sys.argv[1])+1
 TIME_FOR_TASK =2*3600 
 TIME_PER_FIT= 20*60
 SEED=i
 METRIC = autosklearn.metrics.mean_squared_error

 print("SEED: "+str(SEED))
 print("TIME PER TASK: "+str(TIME_FOR_TASK))
 print("TIME PER FIT: "+str(TIME_PER_FIT))
 def get_random_search_object_callback(
        scenario_dict,
        seed,
        ta,
        ta_kwargs,
        metalearning_configurations,
        n_jobs,
        dask_client
 ):
    """Random search."""

    if n_jobs > 1 or (dask_client and len(dask_client.nthreads()) > 1):
        raise ValueError("Please make sure to guard the code invoking Auto-sklearn by "
                         "`if __name__ == '__main__'` and remove this exception.")

    scenario_dict['minR'] = len(scenario_dict['instances'])
    scenario_dict['initial_incumbent'] = 'RANDOM'
    scenario = Scenario(scenario_dict)
    return ROAR(
        scenario=scenario,
        rng=seed,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        run_id=seed,
        dask_client=dask_client,
        n_jobs=n_jobs,
    )



 automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task = TIME_FOR_TASK,\
 per_run_time_limit = TIME_PER_FIT, seed=SEED,\
 resampling_strategy= 'holdout',\
 resampling_strategy_arguments ={'shuffle': False},\
 metric = METRIC,
 get_smac_object_callback=get_random_search_object_callback,
 initial_configurations_via_metalearning=0,
 include_estimators = ['libsvm_svr'],
 )
 automl.fit(X_trainlist, y_trainlist)
 automl.refit(X_trainlist, y_trainlist)
 print(automl.sprint_statistics())
 print(automl.show_models())
 import math
 predictions = automl.predict(X_trainlist)
 score = sklearn.metrics.mean_squared_error(y_trainlist, predictions)
 MAE = sklearn.metrics.mean_absolute_error(y_trainlist, predictions)
 RMSEscore = math.sqrt(score)
 R2 = sklearn.metrics.r2_score(y_trainlist, predictions)
 print("ERRORS TRAIN SET:")
 print('RMSE: ', str(RMSEscore))
 print('MAE: ', str(MAE))
 print('R2:', str(R2))
 predictions = automl.predict(X_testlist)
 score = sklearn.metrics.mean_squared_error(y_testlist, predictions)
 MAE = sklearn.metrics.mean_absolute_error(y_testlist, predictions)
 RMSEscore = math.sqrt(score)
 R2 = sklearn.metrics.r2_score(y_testlist, predictions)
 print("ERRORS TEST SET:")
 print ('RMSE: ', str(RMSEscore))
 print ('MAE: ', str(MAE))
 print('R2:', str(R2))

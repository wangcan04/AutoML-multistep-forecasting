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
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, select_features

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

    # Forecasting window size. Since it takes very long time to extract features with window_size = 200,
    # we extract the feature first.
    window_size = 200

    # The data points skip window. The first 200 data points will be skipped
    # to make sure every point that being forecasted has enough input data. 
    windowsize = 200
    X = data

    # select useful features from training set, assume test set is untouchable
    X_trainlist = X[:int((len(X))*0.67)]
    df_shift, y_shift = make_forecasting_frame(X_trainlist, kind="value", max_timeshift=window_size, rolling_direction=1)
    # since extracting features with window_size = 200 is very time consuming, we do it before search for the models.    df_shift, y_shift = make_forecasting_frame(X, kind="data", max_timeshift=window_size, rolling_direction=1)
    X_feature = extract_features(df_shift, column_id="id", column_sort="time",column_value="value")
    X_feature = impute(X_feature)
    X_feature.index = range(int(len(X_feature)))
    y_shift.index = range(len(y_shift))
    X_feature = X_feature[windowsize-1:]
    y_shift = y_shift[windowsize-1:]
    X_feature = select_features(X_feature, y_shift)
    selected = X_feature.columns.tolist()
 
    # extract features from the whole dataset
    df_shift, y_shift = make_forecasting_frame(X, kind="value", max_timeshift=window_size, rolling_direction=1)
    X_feature = extract_features(df_shift, column_id="id", column_sort="time",column_value="value")
    X_feature = impute(X_feature)
    X_feature.index = range(int(len(X_feature)))
    X_feature = X_feature[windowsize-1:]
    X = X_feature.loc[:, selected]

    y = data[windowsize:]
    X_trainlist = X[:int((len(X))*0.67)]
    y_trainlist = y[:int((len(X))*0.67)]
    X_testlist = X[int((len(X))*0.67):]
    y_testlist = y[int((len(X))*0.67):]

    i=int(sys.argv[1])+1
    TIME_FOR_TASK = 2*60*60 
    TIME_PER_FIT= 20*60
    SEED=i
    METRIC = autosklearn.metrics.mean_squared_error


    print("SEED: "+str(SEED))
    print("TIME PER TASK: "+str(TIME_FOR_TASK))
    print("TIME PER FIT: "+str(TIME_PER_FIT))

    automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task = TIME_FOR_TASK, per_run_time_limit = TIME_PER_FIT, seed=SEED,\
    resampling_strategy='holdout',\
    resampling_strategy_arguments={'shuffle':False},\
    # since we have done the feature extraction, we do not need to do it again here. 
    include_preprocessors = ['no_preprocessing'],\
    metric = METRIC,
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











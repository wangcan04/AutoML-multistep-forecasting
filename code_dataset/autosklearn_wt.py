#!/usr/bin/env python


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,     UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.feature_preprocessing
from autosklearn.pipeline.components.base     import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA,     UNSIGNED_DATA
from autosklearn.util.common import check_none

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


from ConfigSpace.configuration_space import ConfigurationSpace
from autosklearn.pipeline.components.base import     AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import *
import sys


class Tsfreshfeature(AutoSklearnPreprocessingAlgorithm):
      def __init__(self, window_size, random_state=None):
          self.window_size = window_size
      def fit(self, X, y=None):
          self.fitted_ = True
          return self
      def transform(self, X):
          import numpy as np
          import pandas as pd
          from tsfresh.utilities.dataframe_functions import roll_time_series
          from tsfresh.utilities.dataframe_functions import make_forecasting_frame
          from tsfresh.utilities.dataframe_functions import impute
          from tsfresh import extract_features, select_features
          X = np.append(X[0],X[1:,-1])
          # tsfresh does not extract features for the last element in the time series, we add an extra element to 
          # make sure tsfresh does not waste any information
          X = np.append(X,X[-1])
          window_size = self.window_size
          df_shift, y_shift = make_forecasting_frame(X, kind="data", max_timeshift=window_size, rolling_direction=1)
          X = extract_features(df_shift, column_id="id", column_sort="time",column_value="value",disable_progressbar = True)
          X = impute(X)
          global selected
          X = X.loc[:, selected]
          X = np.array(X)
          #skip the first windowsize-1 points
          X = X[199:,:]
          return X
      @staticmethod
      def get_properties(dataset_properties=None):
          return {'shortname': 'window size',
                  'name': 'window size',
                  'handles_regression': True,
                  'handles_classification': False,
                  'handles_multiclass': False,
                  'handles_multilabel': False,
                  'handles_multioutput': False,
                  'is_deterministic': True,
                  'input': (DENSE, UNSIGNED_DATA,SIGNED_DATA),
                  'output': (DENSE, UNSIGNED_DATA,SIGNED_DATA)}
      @staticmethod
      def get_hyperparameter_search_space(dataset_properties=None):
          cs = ConfigurationSpace()
          window_size = UniformIntegerHyperparameter(name = "window_size", lower = 2,upper = 200, default_value=10)
          cs.add_hyperparameters([window_size])
          return cs
autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(Tsfreshfeature)

import numpy as np
import os
import pandas as pd


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
windowsize = 200

X=data
# select useful features from training set, assume test set is untouchable. Since the useful features for different
# window_size on training set and test set may different, we need to predefine the features that been selected.
window_size=50
X_trainlist = X[:int((len(X))*0.67)]
df_shift, y_shift = make_forecasting_frame(X_trainlist, kind="value", max_timeshift=window_size, rolling_direction=1)
X_feature = extract_features(df_shift, column_id="id", column_sort="time",column_value="value")
X_feature = impute(X_feature)
X_feature.index = range(int(len(X_feature)))
y_shift.index = range(len(y_shift))
X_feature = X_feature[windowsize-1:]
y_shift = y_shift[windowsize-1:]
X_feature = select_features(X_feature, y_shift)
selected = X_feature.columns.tolist()

#make training set and test set
y = data
y = y[windowsize:]
X = []
for i in range(len(data)-windowsize):
    x = data[i:i+windowsize]
    X.append(x)
X = np.array(X)

X_train = X[:int(len(X)*0.67)]
y_train = y[:int(len(y)*0.67)]
X_test = X[int(len(X)*0.67):]
y_test = y[int(len(y)*0.67):]






i=int(sys.argv[1])+1
TIME_FOR_TASK = 2*60*60
TIME_PER_FIT =20*60
SEED=i
METRIC = autosklearn.metrics.mean_squared_error


print("SEED: "+str(SEED))
print("TIME PER TASK: "+str(TIME_FOR_TASK))
print("TIME PER FIT: "+str(TIME_PER_FIT))
import autosklearn.regression
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=TIME_FOR_TASK,per_run_time_limit=TIME_PER_FIT,\
    include_preprocessors=['Tsfreshfeature'],\
    resampling_strategy='holdout',\
    resampling_strategy_arguments = {'shuffle':False},\
    metric = METRIC,
)

automl.fit(X_train, y_train)

print(automl.sprint_statistics())
print(automl.show_models())

import math
predictions = automl.predict(X_train)


score = sklearn.metrics.mean_squared_error(y_train, predictions)
MAE = sklearn.metrics.mean_absolute_error(y_train, predictions)
RMSEscore = math.sqrt(score)
R2 = sklearn.metrics.r2_score(y_train, predictions)
print("ERRORS TRAIN SET:")
print('RMSE: ', str(RMSEscore))
print('MAE: ', str(MAE))
print('R2', str(R2))




predictions = automl.predict(X_test)
score = sklearn.metrics.mean_squared_error(y_test, predictions)
MAE = sklearn.metrics.mean_absolute_error(y_test, predictions)
RMSEscore = math.sqrt(score)
R2 = sklearn.metrics.r2_score(y_test, predictions)
print("ERRORS TEST SET:")
print ('RMSE: ', str(RMSEscore))
print ('MAE: ', str(MAE))
print('R2:', str(R2))


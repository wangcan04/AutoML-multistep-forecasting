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
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.feature_preprocessing
from autosklearn.pipeline.components.base \
    import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, \
    UNSIGNED_DATA
from autosklearn.util.common import check_none

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class Window_size(AutoSklearnPreprocessingAlgorithm):
        def __init__(self, window_size, random_state=None):
            self.window_size = window_size
        def fit(self, X, y=None):
            self.fitted_ = True
            return self
        def transform(self, X):
            import numpy as np
            import pandas as pd
            window_size = self.window_size
            X = X[:,-window_size:]
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
autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(Window_size)
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

 import numpy as np
 import os
 import pandas as pd
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
 X_trainlist = X[:int(len(X)*0.67)]
 y_trainlist = y[:int(len(X)*0.67)]
 X_testlist = X[int(len(X)*0.67):]
 y_testlist = y[int(len(X)*0.67):]


 i=int(sys.argv[1])+1
 TIME_FOR_TASK = 2*3600
 TIME_PER_FIT = 20*60
 SEED=i
 METRIC = autosklearn.metrics.mean_squared_error


 print("SEED: "+str(SEED))
 print("TIME PER TASK: "+str(TIME_FOR_TASK))
 print("TIME PER FIT: "+str(TIME_PER_FIT))
 automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task = TIME_FOR_TASK,per_run_time_limit = TIME_PER_FIT,\
 include_preprocessors=['Window_size'],\
 seed=SEED, \
 resampling_strategy= 'holdout',\
 resampling_strategy_arguments ={'shuffle': False},\
 metric = METRIC,\
 ensemble_size=1
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





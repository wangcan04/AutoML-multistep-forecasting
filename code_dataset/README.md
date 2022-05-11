# AutoML-timeseries

AutoML-timeseries is an automated model for time series forecasting, including automated window size selection and automated feature extraction. Here we show two kinds of baselines, including simple baselines(ARIMA, Moving average, Support vector machine, and Random forest) and AutoML baselines (Auto-Keras, and auto-sklearn), and our AutoML variants for time series forecasting. 

## Requirements
some python packages are needed:

auto-sklearn  0.8.0 

tsfresh  0.16.1 

autokeras  1.0.12

pmdarima  1.8.0 


## Simple baselines
4 simple stastical models and machine learning models are used.

ARIMA: 
* arima.py

Moving average: 
* mv1.py 
* mv10.py 
* mv200.py

SVM: 
* svm10.py 
* svm200.py

RF:
 * rf10.py 
 * rf200.py

Random seed for SVM, RF experiments is all 10.
## AutoML baselines
Auto-Keras: 
* autokeras10.py 
* autokeras200.py

The number of models that can be evaluated with 2 hours time budget for every dataset is in autokerasmodelcount.txt

Auto-sklearn: 
* autosklearn10.py 
* autosklearn200.py 


## AutoML variants for time series forecasting:
Auto-sklearn with automated window size selection (W):
* autosklearn_w_ens.py
* autosklearn_w_single.py

Auto-sklearn with tsfresh features (T):
* autosklearn_t10.py
* autosklearn_t200.py

Auto-sklearn with automated window size selection and tsfresh features (WT):
* autosklearn_wt.py

Random seeds for autosklearn experiments are 1 to 25.


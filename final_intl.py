# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 08:43:21 2020

@author: Group 3
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import datetime
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
import seaborn as sns
import statistics as sts  # to implement mode function
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time

d=pd.read_csv("C:\\datascience\\Airfare_Prediction_Project\\data_sets\\international.csv",index_col ='InvoiceDate',parse_dates = True)
train=d[0:374]
test=d[374:]


###############             Holt's winter exponential method            #################################
hwe_model_add_add = ExponentialSmoothing(d["NetFare"],seasonal="add",seasonal_periods=31).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0],end = test.index[-1])

# Creating a pickle file for the intl fare prediction
filename = 'intl_air-fare-prediction-model.pkl'
pickle.dump(hwe_model_add_add, open(filename, 'wb'))


test['holt_winter']=pred_hwe_add_add
rmse_ht_winter= np.sqrt(mean_squared_error(test['NetFare'],test['holt_winter']))
rmse_ht_winter
test[['NetFare','holt_winter']].plot(figsize=(10,5))




######################          ARIMA           ####################################################
from statsmodels.tsa.arima_model import ARIMA
model_intl = ARIMA(d, order=(0,1,1))
model_fit=model_intl.fit()
test['arima']=model_fit.forecast(steps=31)[0]

d['NetFare'].plot(figsize = (16,5), legend=True)
test['arima'].plot(figsize = (16,5),legend = True);
test[['NetFare','arima']].plot(figsize=(10,5))

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
mean_absolute_error(test['NetFare'],test['arima'])
np.sqrt(mean_squared_error(test['NetFare'],test['arima']))




######################################      AUTO -ARIMA         #################################

from pmdarima import auto_arima
auto_arima_model = auto_arima(d['NetFare'],start_p=0,
                              start_q=0,max_p=1,max_q=1,
                              m=31,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=True)

auto_arima_model.predict_in_sample( )
pred_test = pd.Series(auto_arima_model.predict(n_periods=31))
pred_test.index = test.index
test['pred_test']=pred_test
test[['NetFare','pred_test']].plot(figsize=(10,5))
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
np.sqrt(mean_squared_error(test['NetFare'],test['pred_test']))
mean_absolute_error(test['NetFare'],test['pred_test'])




pred_test = auto_arima_model.predict(5)




############################            SARIMAX         ############################################
best_fit_model = sm.tsa.statespace.SARIMAX(d["NetFare"],
                                                     order = (1,1,1),seasonal_order = (1,1,1,31)).fit()
srma_pred = best_fit_model.predict(start =test.index[0],end = test.index[-1])

test['PRED']=srma_pred
test[['NetFare','PRED']].plot(figsize=(10,5))
test['pred']=model_fit.forecast(steps=31)[0]

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
mean_absolute_error(test['NetFare'],test['PRED'])
np.sqrt(mean_squared_error(test['NetFare'],test['PRED']))







##############################          LSTM model          ######################################
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
train_sc = scaler.transform(train)
test_sc = scaler.transform(test)

from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 10
n_features= 1
generator = TimeseriesGenerator(train_sc, train_sc, length=n_input, batch_size=1)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()
history = lstm_model.fit_generator(generator,epochs=50)

lstm_predictions_scaled = list()

batch = train_sc[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
    
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
#lstm_predictions
test['LSTM_Predictions'] = lstm_predictions
test['NetFare'].plot(figsize = (16,5), legend=True)
test['LSTM_Predictions'].plot(legend = True);
rmse_lstm = np.sqrt(mean_squared_error(test['NetFare'],test['LSTM_Predictions'])) 
mae_lstm = mean_absolute_error(test['NetFare'],test['LSTM_Predictions']) 




##MODEL BASED APPROACH

## importing the libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as snc
import scipy
from sklearn import metrics

##importing the dataset
data=pd.read_csv("C:\\datascience\\Airfare_Prediction(Project)\\data_sets\\Concatenate_B2C_B2E.csv")


data_air= data[data['ProductType']=="Air"]
data.columns
data.dtypes

# FEATURE ENGINEERING
#Segrigating  the 'InvoiceDate' column into each parts
data_air['InvoiceDate'] =pd.to_datetime(data_air['InvoiceDate'])
data_air['day'] = pd.DatetimeIndex(data_air['InvoiceDate']).day
data_air['month'] = pd.DatetimeIndex(data_air['InvoiceDate']).month
data_air['year'] = pd.DatetimeIndex(data_air['InvoiceDate']).year
# data_air['hour'] = pd.DatetimeIndex(data_air['InvoiceDate']).hour
# data_air['minute']= pd.DatetimeIndex(data_air['InvoiceDate']).minute
data_air['weekday'] = pd.DatetimeIndex(data_air['InvoiceDate']).weekday
data_air['NetFare']=data_air['NetFare'].apply(pd.to_numeric,errors='coerce')

##checking the null value
data_air.isnull().sum()

data_air['ItineraryType'].unique()


## So we have 2 variables in the iteration type, i.e we have 2 type of flights of which we need to predict the airfare

data_airInt = data_air[data_air['ItineraryType']=='International']

encoded_columns = pd.get_dummies(data_airInt['ProductType'])
data_airInt = data_airInt.join(encoded_columns).drop('ProductType', axis=1)

encoded_columns = pd.get_dummies(data_airInt['ItineraryType'])
data_airInt = data_airInt.join(encoded_columns).drop('ItineraryType', axis=1)
data_airInt.dtypes
data_airInt1=data_airInt.groupby(pd.Grouper(freq='d')).mean()
data_airInt1 = data_airInt.groupby(pd.Grouper(key='InvoiceDate',freq='d')).mean()
data_airInt1.dropna(inplace=True)

data_airInt1=data_airInt1[(data_airInt1['NetFare'] < 48000) & (data_airInt1['NetFare'] > 12000)]


##Segrigating the predictors and target columns
x=data_airInt1.iloc[:,1:]
y=data_airInt1.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=42, test_size=0.3)

##########################          LINEAR REGRESSION           ####################################

from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(x_train,y_train)

model1.coef_
model1.intercept_

model1.score(x_train,y_train) ##R2=0.179
model1.score(x_test,y_test) ## R2=0.05

from sklearn.model_selection import cross_val_score
score=cross_val_score(model1,x,y,cv=5)

score.mean()##0.113
y_pred=model1.predict(x_test)
print('MAE:', mean_absolute_error(y_test, y_pred))##7051
print('MSE:', mean_squared_error(y_test, y_pred))##80497718
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))##8972




############################            LASSO REGRESSION            ######################################

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
model2=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,60,50,70,80,90,100,120,140]}
lasso_regressor=GridSearchCV(model2,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(x_train,y_train)
model2.score(x_train,y_train) ##R2=0.143
model2.score(x_test,y_test)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

y_pred=lasso_regressor.predict(x_test)
snc.distplot(y_test-y_pred)
plt.scatter(y_test,y_pred)


print('MAE:', metrics.mean_absolute_error(y_test,y_pred))#7084
print('MSE:', metrics.mean_squared_error(y_test, y_pred))#80938183
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))#8996



########################            DECISION TREE REGRESSOR             ###############################

from sklearn.tree import DecisionTreeRegressor
model3=DecisionTreeRegressor(criterion= "mse")
model3.fit(x_train,y_train)
model3.score(x_train,y_train)##r^2=0.978
# model3.score(x_test,y_test)##r^2= 0.04

from sklearn.model_selection import cross_val_score
score=cross_val_score(model3,x,y,cv=5)
score.mean()

y_pred=model3.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))#176305808
print('Test Variance score: %.2f' % r2_score(y_test, y_pred))#-1.06
DecisionTreeRegressor()
## Hyper Parameter Optimization

params={
  "splitter"    : ["best","random"] ,
  "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
  "min_samples_leaf" : [ 1,2,3,4,5 ],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
  "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70]
    
}
## Hyperparameter optimization using GridSearchCV
from sklearn.model_selection import GridSearchCV
random_search=GridSearchCV(model3,param_grid=params,scoring='neg_mean_squared_error',n_jobs=-1,cv=2,verbose=3)

random_search.fit(x,y)
random_search.best_params_

random_search.best_score_

y_pred=random_search.predict(x_test)

snc.distplot(y_test-y_pred)

#Model evalution
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))#6811
print('MSE:', metrics.mean_squared_error(y_test, y_pred))#75209654
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))#8672


#########################           RANDOM FOREST REGRESSOR             ###############################

#First applying for domestic .same  can be repeated for international  also
from sklearn.ensemble import RandomForestRegressor
model_dom= RandomForestRegressor()
model_dom.fit(x_train,y_train)
model_dom.score(x_train,y_train)##R^2=0.99
model_dom.score(x_test,y_test)#R^2=0.99

##Now lets go for Hypertuning approach to get the parameters
# Number of trees in random forest
n_estimators = [200]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True]
max_depth = [None]
min_weight_fraction_leaf=[0]
max_leaf_nodes=[None]
min_impurity_decrease=[0]

# Create the random grid
random_grid = {'max_features': max_features,
               'bootstrap': bootstrap,
               'n_estimators': n_estimators,
               'max_features' : max_features,
               'min_samples_split' : min_samples_split,
               'max_depth' : max_depth,
               'min_weight_fraction_leaf' : min_weight_fraction_leaf,
               'max_leaf_nodes': max_leaf_nodes,
               'min_impurity_decrease':min_impurity_decrease
               }
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
# Fit the random search model

from sklearn.model_selection import RandomizedSearchCV

estimator = RandomForestRegressor()
n_jobs=4
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 3, verbose=2, random_state=42, n_jobs = 8)
rf_random.fit(x_train, y_train)
print(rf_random.best_params_)
y_pred = rf_random.best_estimator_.predict(x_test)


##Model evalution
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))##7486
print('MSE:', metrics.mean_squared_error(y_test, y_pred))##93879228
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))##9689





#############################           XGBOOST REGRESSOR           ################################

#First applying for domestic .same  can be repeated for international  also
import xgboost as xgb
regressor=xgb.XGBRegressor()

model_dom= xgb.XGBRegressor()
model_dom.fit(x_train,y_train)
model_dom.score(x_train,y_train)
model_dom.score(x_test,y_test)

##Now lets go for Hypertuning approach to get the parameters
# Number of trees in random forest
n_estimators = [200]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True]
max_depth = [None]
min_weight_fraction_leaf=[0]
max_leaf_nodes=[None]
min_impurity_decrease=[0]

# Create the random grid
random_grid = {'max_features': max_features,
               'bootstrap': bootstrap,
               'n_estimators': n_estimators,
               'max_features' : max_features,
               'min_samples_split' : min_samples_split,
               'max_depth' : max_depth,
               'min_weight_fraction_leaf' : min_weight_fraction_leaf,
               'max_leaf_nodes': max_leaf_nodes,
               'min_impurity_decrease':min_impurity_decrease
               }
regressor=xgb.XGBRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
# Fit the random search model

from sklearn.model_selection import RandomizedSearchCV

estimator = xgb.XGBRegressor()
n_jobs=4
rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid, cv = 3, verbose=2, random_state=42, n_jobs = 8)
rf_random.fit(x_train, y_train)
print(rf_random.best_params_)
y_pred = rf_random.best_estimator_.predict(x_test)

#Model evalution
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))#8467
print('MSE:', metrics.mean_squared_error(y_test, y_pred))#119350311
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))#10924


#########################           KNN REGRESSOR           ########################################

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

regressor=KNeighborsRegressor(n_neighbors=1)
regressor.fit(x_train,y_train)



accuracy_rate = []
# Will take some time
for i in range(1,40):
    
    knn = KNeighborsRegressor(n_neighbors=i)
    score=cross_val_score(knn,x,y,cv=4,scoring="neg_mean_squared_error")
    accuracy_rate.append(score.mean())
    




import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
          markerfacecolor='red', markersize=10)
#plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
  #        markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')




knn = KNeighborsRegressor(n_neighbors=1)

knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))##8508
print('MSE:', metrics.mean_squared_error(y_test, y_pred))##121222444
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))##11010




##############################          ANN             ##############################################

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

# Fitting the ANN to the Training set
model_history=NN_model.fit(x_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 10)

y_pred=NN_model.predict(x_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))##8357
print('MSE:', metrics.mean_squared_error(y_test, y_pred))##123832399
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))##11128



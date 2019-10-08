#imports
#numpy,pandas,scipy, math, matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statistics as st
import math
from math import sqrt
import matplotlib.pyplot as plt
import plotly.express as px
import metrics
import re
 #estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
#model metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
#cross validation
from sklearn.model_selection import train_test_split

%run "C:/Users/andre/Desktop/Ubiqum/IoT analytics/Task 1/Wifi/GOOD/GOOD MODELS/Preprocessing/WIFI_PREPROCESSING1.py

# =============================================================================
# MODEL FOR BUILDING 
# =============================================================================

# #Splitting data into train and test

x_train = trainingDatawoduplicates.iloc[:, np.r_[0:466]]
y_train = trainingDatawoduplicates.iloc[: , 466:470]
x_test  = validationwoduplicates.iloc[:, np.r_[0:466]]
y_test  = validationwoduplicates.iloc[: , 466:470]

#KNN
KNN_building = KNeighborsClassifier(n_neighbors=10).fit(x_train, y_train['BUILDINGID'])
KNN_building.score(x_test, y_test['BUILDINGID'])
#KNNbuild_cv_score = cross_val_score(KNN_building, x_test, y_test['BUILDINGID'], cv=10)
KNNpred_building = KNN_building.predict(x_test)
print(classification_report(y_test['BUILDINGID'], KNNpred_building))


# =============================================================================
# MODEL FOR FLOOR
# =============================================================================

x_train['BUILDING'] = y_train['BUILDINGID']
x_test['BUILDING'] = KNNpred_building

builddummytrain = pd.get_dummies(x_train['BUILDING'])
builddummytest = pd.get_dummies(x_test['BUILDING'])

x_train = pd.concat([x_train, builddummytrain], axis=1)
x_test = pd.concat([x_test, builddummytest], axis=1)

del x_train['BUILDING']
del x_test['BUILDING']

#KNN
KNN_floor = KNeighborsClassifier(n_neighbors=10)
KNN_floor.fit(x_train, y_train['FLOOR'])
KNN_floor.score(x_test, y_test['FLOOR'])
#KNNfloor_cv_score = cross_val_score(KNN_building, x_test, y_test['FLOOR'], cv=10)
KNNpred_floor = KNN_floor.predict(x_test)
print(classification_report(y_test['FLOOR'], KNNpred_floor))
confusion_matrix(y_test['FLOOR'], KNNpred_floor)

print("Accuracy:", accuracy_score(y_test['FLOOR'], KNNpred_floor))
print("Kappa:",cohen_kappa_score(y_test['FLOOR'], KNNpred_floor))


# =============================================================================
# MODELLING FOR LONGITUDE
# =============================================================================
modelKNN_long = KNeighborsRegressor(n_neighbors=10)
modelKNN_long.fit(x_train,y_train['LONGITUDE'])
modelKNN_long_pred = modelKNN_long.predict(x_test)
round(modelKNN_long.score(x_test,y_test['LONGITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LONGITUDE'], modelKNN_long_pred)))
print("MAE:", mean_absolute_error(y_test['LONGITUDE'], modelKNN_long_pred))
print("R2:",r2_score(y_test['LONGITUDE'], modelKNN_long_pred))

# =============================================================================
# MODELLING FOR LATITUDE
# =============================================================================

x_train['LONGITUDE'] = y_train['LONGITUDE']
x_test['LONGITUDE'] = modelKNN_long_pred

modelKNN_lat = KNeighborsRegressor(n_neighbors=10)
modelKNN_lat.fit(x_train,y_train['LATITUDE'])
modelKNN_lat_pred = modelKNN_lat.predict(x_test)
round(modelKNN_lat.score(x_test,y_test['LATITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LATITUDE'], modelKNN_lat_pred)))
print("MAE:", mean_absolute_error(y_test['LATITUDE'], modelKNN_lat_pred))
print("R2:",r2_score(y_test['LATITUDE'], modelKNN_lat_pred))

# error analysis

results = pd.DataFrame()
results["BUILDINGID"] = KNNpred_building
results["LATITUDE"]  = modelKNN_lat_pred
results["LONGITUDE"] = modelKNN_long_pred
results["FLOOR"] = KNNpred_floor
results["type"] = "pred"
res = pd.DataFrame()
res["BUILDINGID"] = y_test['BUILDINGID']
res["LATITUDE"]  = y_test['LATITUDE']
res["LONGITUDE"] = y_test['LONGITUDE']
res["FLOOR"] = y_test['FLOOR']
res["type"] = "real"
####
fresults = results.append(res)
####
results["AE_Long"] = abs(modelKNN_long_pred - y_test['LONGITUDE'])
results["RE_Long"] = (abs(modelKNN_long_pred - y_test['LONGITUDE']))/abs(y_test['LONGITUDE'])
results["AE_Lat"] = abs(modelKNN_lat_pred -y_test['LATITUDE'])
results["RE_Lat"] = abs((modelKNN_lat_pred - y_test['LATITUDE']))/y_test['LATITUDE']

print("MRE Lat:",np.mean(results["RE_Lat"]))
print("MRE Long:",np.mean(results["RE_Long"]))

plt.hist(results["AE_Lat"])
plt.hist(results["AE_Long"])
plt.scatter(x = results["LATITUDE"], y = results["AE_Lat"])
plt.scatter(x = results["LONGITUDE"], y = results["AE_Long"])

fig = px.histogram(results, x="AE_Lat")
fig.show()
fig = px.histogram(results, x="AE_Long")
fig.show()
fig = px.scatter(results, x="LATITUDE", y="RE_Lat")
fig.show()
fig = px.scatter(results, x="LONGITUDE", y="RE_Long")
fig.show()


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
train   = pd.concat([x_train, y_train['BUILDINGID']], axis=1)
test    = pd.concat([x_test, y_test['BUILDINGID']], axis=1)

# configure bootstrap KNN
n_iterations = 100
buildingaccuracyscores = []
buildingkappascores    = []
buildingpredictions    = pd.DataFrame()

for i in range(n_iterations):
    # prepare train and test sets
    trains = resample(train, n_samples = int(len(train) * 0.50))
    tests  = resample(test, n_samples = int(len(test) * 0.50))
    # fit model
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(trains.iloc[:,:-1], trains.iloc[:,-1])
    # evaluate model
    predictions = model.predict(tests.iloc[:,:-1])
    score1 = accuracy_score(tests.iloc[:,-1], predictions)
    score2 = cohen_kappa_score(tests.iloc[:,-1], predictions)
    buildingaccuracyscores.append(score1)
    buildingkappascores.append(score2)
    buildingpredictions = buildingpredictions.append(pd.DataFrame(predictions), ignore_index=True)

plt.hist(buildingaccuracyscores)
plt.hist(buildingkappascores)


alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lowerbuildacc = max(0.0, np.percentile(buildingaccuracyscores, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperbuildacc = min(1.0, np.percentile(buildingaccuracyscores, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lowerbuildacc*100, upperbuildacc*100))

alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lowerbuildk = max(0.0, np.percentile(buildingkappascores, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperbuildk = min(1.0, np.percentile(buildingkappascores, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lowerbuildacc*100, upperbuildacc*100))

#KNN
KNN_building = KNeighborsClassifier(n_neighbors=10).fit(x_train, y_train['BUILDINGID'])
KNN_building.score(x_test, y_test['BUILDINGID'])
#KNNbuild_cv_score = cross_val_score(KNN_building, x_test, y_test['BUILDINGID'], cv=10)
KNNpred_building = KNN_building.predict(x_test)
print(classification_report(y_test['BUILDINGID'], KNNpred_building))
print(confusion_matrix(y_test['BUILDINGID'], KNNpred_building))
# configure bootstrap RF
n_iterations = 100
buildingaccuracyscoresrf = []
buildingkappascoresrf    = []
buildingpredictionsrf    = pd.DataFrame()

for i in range(n_iterations):
    # prepare train and test sets
    trains = resample(train, n_samples = int(len(train) * 0.50))
    tests  = resample(test, n_samples = int(len(test) * 0.50))
    # fit model
    model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
    model.fit(trains.iloc[:,:-1], trains.iloc[:,-1])
    # evaluate model
    predictions = model.predict(tests.iloc[:,:-1])
    score1 = accuracy_score(tests.iloc[:,-1], predictions)
    score2 = cohen_kappa_score(tests.iloc[:,-1], predictions)
    buildingaccuracyscoresrf.append(score1)
    buildingkappascoresrf.append(score2)
    buildingpredictionsrf = buildingpredictionsrf.append(pd.DataFrame(predictions), ignore_index=True)


alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lowerbuildacc = max(0.0, np.percentile(buildingaccuracyscoresrf, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperbuildacc = min(1.0, np.percentile(buildingaccuracyscoresrf, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lowerbuildacc*100, upperbuildacc*100))

alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lowerbuildk = max(0.0, np.percentile(buildingkappascoresrf, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperbuildk = min(1.0, np.percentile(buildingkappascoresrf, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lowerbuildk*100, upperbuildk*100))

RFbuilding = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_train, y_train['BUILDINGID'])
RFbuilding.score(x_test, y_test['BUILDINGID'])
# 10-Fold Cross validation
#RFCbuild_cv_score = cross_val_score(RFbuilding, x, y, cv=10)
RFCpred_build = RFbuilding.predict(x_test)
print(classification_report(y_test['BUILDINGID'], RFCpred_build))

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

train   = pd.concat([x_train, y_train['FLOOR']], axis=1)
test   = pd.concat([x_test, y_test['FLOOR']], axis=1)

# configure bootstrap KNN
n_iterations = 100
flooraccuracyscores = []
floorkappascores    = []
floorpredictions    = pd.DataFrame()

for i in range(n_iterations):
    # prepare train and test sets
    trains = resample(train, n_samples = int(len(train) * 0.50))
    tests  = resample(test, n_samples = int(len(test) * 0.50))
    # fit model
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(trains.iloc[:,:-1], trains.iloc[:,-1])
    # evaluate model
    predictions = model.predict(tests.iloc[:,:-1])
    score1 = accuracy_score(tests.iloc[:,-1], predictions)
    score2 = cohen_kappa_score(tests.iloc[:,-1], predictions)
    flooraccuracyscores.append(score1)
    floorkappascores.append(score2)
    floorpredictions = floorpredictions.append(pd.DataFrame(predictions), ignore_index=True)

plt.hist(flooraccuracyscores)
plt.hist(floorkappascores)

alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lowerflooracc = max(0.0, np.percentile(flooraccuracyscores, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperflooracc = min(1.0, np.percentile(flooraccuracyscores, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lowerflooracc*100, upperflooracc*100))

alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lowerfloork = max(0.0, np.percentile(floorkappascores, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperfloork = min(1.0, np.percentile(floorkappascores, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lowerfloork*100, upperfloork*100))


#KNN
KNN_floor = KNeighborsClassifier(n_neighbors=1)
KNN_floor.fit(x_train, y_train['FLOOR'])
KNN_floor.score(x_test, y_test['FLOOR'])
#KNNfloor_cv_score = cross_val_score(KNN_building, x_test, y_test['FLOOR'], cv=10)
KNNpred_floor = KNN_floor.predict(x_test)
print(classification_report(y_test['FLOOR'], KNNpred_floor))
confusion_matrix(y_test['FLOOR'], KNNpred_floor)

print("Accuracy:", accuracy_score(y_test['FLOOR'], KNNpred_floor))
print("Kappa:",cohen_kappa_score(y_test['FLOOR'], KNNpred_floor))

# configure bootstrap RF
n_iterations = 100
flooraccuracyscoresrf = []
floorkappascoresrf    = []
floorpredictionsrf    = pd.DataFrame()

for i in range(n_iterations):
    # prepare train and test sets
    trains = resample(train, n_samples = int(len(train) * 0.50))
    tests  = resample(test, n_samples = int(len(test) * 0.50))
    # fit model
    model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
    model.fit(trains.iloc[:,:-1], trains.iloc[:,-1])
    # evaluate model
    predictions = model.predict(tests.iloc[:,:-1])
    score1 = accuracy_score(tests.iloc[:,-1], predictions)
    score2 = cohen_kappa_score(tests.iloc[:,-1], predictions)
    flooraccuracyscoresrf.append(score1)
    floorkappascoresrf.append(score2)
    floorpredictionsrf = floorpredictions.append(pd.DataFrame(predictions), ignore_index=True)

plt.hist(flooraccuracyscoresrf)
plt.hist(floorkappascoresrf)

alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lowerflooraccrf = max(0.0, np.percentile(flooraccuracyscoresrf, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperflooraccrf = min(1.0, np.percentile(flooraccuracyscoresrf, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lowerflooraccrf*100, upperflooraccrf*100))

alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lowerfloorkrf = max(0.0, np.percentile(floorkappascoresrf, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upperfloorkrf = min(1.0, np.percentile(floorkappascoresrf, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lowerfloorkrf*100, upperfloorkrf*100))

#RF
RFfloor = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_train, y_train['FLOOR'])
RFfloor.score(x_test, y_test['FLOOR'])
# 10-Fold Cross validation
#RFCbuild_cv_score = cross_val_score(RFbuilding, x, y, cv=10)
RFCpred_floor = RFfloor.predict(x_test)
print(classification_report(y_test['FLOOR'], RFCpred_floor))
confusion_matrix(y_test['FLOOR'], RFCpred_floor)

print("Accuracy:", accuracy_score(y_test['FLOOR'], RFCpred_floor))
print("Kappa:",cohen_kappa_score(y_test['FLOOR'], RFCpred_floor))
# =============================================================================
# MODELLING FOR LONGITUDE
# =============================================================================

train   = pd.concat([x_train, y_train['LONGITUDE']], axis=1)
test   = pd.concat([x_test, y_test['LONGITUDE']], axis=1)

# configure bootstrap KNN

n_iterations = 100
longRMSE = []
longMAE  = []
longR2   = []
longpredictions    = pd.DataFrame()

for i in range(n_iterations):
    # prepare train and test sets
    trains = resample(train, n_samples = int(len(train) * 0.50))
    tests  = resample(test, n_samples = int(len(test) * 0.50))
    # fit model
    model = KNeighborsRegressor(n_neighbors=14)
    model.fit(trains.iloc[:,:-1], trains.iloc[:,-1])
    # evaluate model
    predictions = model.predict(tests.iloc[:,:-1])
    score1 = sqrt(mean_squared_error(tests.iloc[:,-1], predictions))
    score2 = mean_absolute_error(tests.iloc[:,-1], predictions)
    score3 = r2_score(tests.iloc[:,-1], predictions)
    longRMSE.append(score1)
    longMAE.append(score2)
    longR2.append(score3)
    longpredictions = longpredictions.append(pd.DataFrame(predictions), ignore_index=True)

plt.hist(longRMSE)
plt.hist(longMAE)
plt.hist(longR2)

devlongRMSE = st.stdev(longRMSE)
devlongMAE = st.stdev(longMAE)
devlongR2 = st.stdev(longR2)

meanlongRMSE = st.mean(longRMSE)
meanlongMAE  = st.mean(longMAE)
meanlongR2   = st.mean(longR2)

print("RMSE is ", meanlongRMSE, "with +/-",devlongRMSE)
print("MAE is ", meanlongMAE, "with +/-",devlongMAE)
print("R2 is ", meanlongR2, "with +/-",devlongR2)

modelKNN_long = KNeighborsRegressor(n_neighbors=14)
modelKNN_long.fit(x_train,y_train['LONGITUDE'])
modelKNN_long_pred = modelKNN_long.predict(x_test)
round(modelKNN_long.score(x_test,y_test['LONGITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LONGITUDE'], modelKNN_long_pred)))
print("MAE:", mean_absolute_error(y_test['LONGITUDE'], modelKNN_long_pred))
print("R2:",r2_score(y_test['LONGITUDE'], modelKNN_long_pred))

# configure bootstrap RF
n_iterations = 100
longRMSErf = []
longMAErf  = []
longR2rf   = []
longpredictionsrf    = pd.DataFrame()

for i in range(n_iterations):
    # prepare train and test sets
    trains = resample(train, n_samples = int(len(train) * 0.50))
    tests  = resample(test, n_samples = int(len(test) * 0.50))
    # fit model
    model = RandomForestRegressor()
    model.fit(trains.iloc[:,:-1], trains.iloc[:,-1])
    # evaluate model
    predictions = model.predict(tests.iloc[:,:-1])
    score1 = sqrt(mean_squared_error(tests.iloc[:,-1], predictions))
    score2 = mean_absolute_error(tests.iloc[:,-1], predictions)
    score3 = r2_score(tests.iloc[:,-1], predictions)
    longRMSErf.append(score1)
    longMAErf.append(score2)
    longR2rf.append(score3)
    longpredictionsrf = longpredictionsrf.append(pd.DataFrame(predictions), ignore_index=True)

plt.hist(longRMSErf)
plt.hist(longMAErf)
plt.hist(longR2rf)

devlongRMSErf = st.stdev(longRMSErf)
devlongMAErf  = st.stdev(longMAErf)
devlongR2rf   = st.stdev(longR2rf)

meanlongRMSErf = st.mean(longRMSErf)
meanlongMAErf  = st.mean(longMAErf)
meanlongR2rf   = st.mean(longR2rf)

print("RMSE is ", meanlongRMSErf, "with +/-",devlongRMSErf)
print("MAE is ", meanlongMAErf, "with +/-",devlongMAErf)
print("R2 is ", meanlongR2rf, "with +/-",devlongR2rf)

modelRF_long = RandomForestRegressor()
modelRF_long.fit(x_train,y_train['LONGITUDE'])
modelRF_long_pred = modelRF_long.predict(x_test)
round(modelRF_long.score(x_test,y_test['LONGITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LONGITUDE'], modelRF_long_pred)))
print("MAE:", mean_absolute_error(y_test['LONGITUDE'], modelRF_long_pred))
print("R2:",r2_score(y_test['LONGITUDE'], modelRF_long_pred))
# =============================================================================
# MODELLING FOR LATITUDE
# =============================================================================
 
x_train['LONGITUDE'] = y_train['LONGITUDE']
x_test['LONGITUDE'] = modelKNN_long_pred

train   = pd.concat([x_train, y_train['LATITUDE']], axis=1)
test   = pd.concat([x_test, y_test['LATITUDE']], axis=1)

# configure bootstrap
n_iterations = 100
 
latRMSE = []
latMAE  = []
latR2   = []
latpredictions    = pd.DataFrame()

for i in range(n_iterations):
    # prepare train and test sets
    trains = resample(train, n_samples = int(len(train) * 0.50))
    tests  = resample(test, n_samples = int(len(test) * 0.50))
    # fit model
    model = KNeighborsRegressor(n_neighbors=10)
    model.fit(trains.iloc[:,:-1], trains.iloc[:,-1])
    # evaluate model
    predictions = model.predict(tests.iloc[:,:-1])
    score1 = sqrt(mean_squared_error(tests.iloc[:,-1], predictions))
    score2 = mean_absolute_error(tests.iloc[:,-1], predictions)
    score3 = r2_score(tests.iloc[:,-1], predictions)
    latRMSE.append(score1)
    latMAE.append(score2)
    latR2.append(score3)
    latpredictions = latpredictions.append(pd.DataFrame(predictions), ignore_index=True)

plt.hist(latRMSE)
plt.hist(latMAE)
plt.hist(latR2)

devlatRMSE = st.stdev(latRMSE)
devlatMAE  = st.stdev(latMAE)
devlatR2   = st.stdev(latR2)

meanlatRMSE = st.mean(latRMSE)
meanlatMAE  = st.mean(latMAE)
meanlatR2   = st.mean(latR2)

print("RMSE is ", meanlatRMSE, "with +/-",devlongRMSE)
print("MAE is ", meanlatMAE, "with +/-",devlongMAE)
print("R2 is ", meanlatR2, "with +/-",devlongR2)

modelKNN_long = KNeighborsRegressor(n_neighbors=8)
modelKNN_long.fit(x_train,y_train['LATITUDE'])
modelKNN_long_pred = modelKNN_long.predict(x_test)
round(modelKNN_long.score(x_test,y_test['LATITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LATITUDE'], modelKNN_long_pred)))
print("MAE:", mean_absolute_error(y_test['LATITUDE'], modelKNN_long_pred))
print("R2:",r2_score(y_test['LATITUDE'], modelKNN_long_pred))

# configure bootstrap
n_iterations = 100
latRMSErf = []
latMAErf  = []
latR2rf   = []
latpredictionsrf    = pd.DataFrame()

for i in range(n_iterations):
    # prepare train and test sets
    trains = resample(train, n_samples = int(len(train) * 0.50))
    tests  = resample(test, n_samples = int(len(test) * 0.50))
    # fit model
    model = RandomForestRegressor()
    model.fit(trains.iloc[:,:-1], trains.iloc[:,-1])
    # evaluate model
    predictions = model.predict(tests.iloc[:,:-1])
    score1 = sqrt(mean_squared_error(tests.iloc[:,-1], predictions))
    score2 = mean_absolute_error(tests.iloc[:,-1], predictions)
    score3 = r2_score(tests.iloc[:,-1], predictions)
    latRMSErf.append(score1)
    latMAErf.append(score2)
    latR2rf.append(score3)
    longpredictions = latpredictionsrf.append(pd.DataFrame(predictions), ignore_index=True)

plt.hist(latRMSErf)
plt.hist(latMAErf)
plt.hist(latR2rf)

devlatRMSErf = st.stdev(latRMSErf)
devlatMAErf  = st.stdev(latMAErf)
devlatR2rf   = st.stdev(latR2rf)

meanlatRMSErf = st.mean(latRMSErf)
meanlatMAErf  = st.mean(latMAErf)
meanlatR2rf   = st.mean(latR2rf)

print("RMSE is ", meanlatRMSErf, "with +/-",devlatRMSErf)
print("MAE is ", meanlatMAErf, "with +/-",devlatMAErf)
print("R2 is ", meanlatR2rf, "with +/-",devlatR2rf)

modelRF_lat = RandomForestRegressor()
modelRF_lat.fit(x_train,y_train['LATITUDE'])
modelRF_lat_pred = modelRF_lat.predict(x_test)
round(modelRF_lat.score(x_test,y_test['LATITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LATITUDE'], modelRF_lat_pred)))
print("MAE:", mean_absolute_error(y_test['LATITUDE'], modelRF_lat_pred))
print("R2:",r2_score(y_test['LATITUDE'], modelRF_lat_pred))

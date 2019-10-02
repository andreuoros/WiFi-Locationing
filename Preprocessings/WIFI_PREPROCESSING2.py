#imports
#numpy,pandas,scipy, math, matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy import stats
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
#model metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
#cross validation
from sklearn.model_selection import train_test_split

trainingData = pd.read_csv("C:/Users/andre/Desktop/Ubiqum/IoT analytics/Task 1/Wifi/trainingData.csv")
validationData = pd.read_csv("C:/Users/andre/Desktop/Ubiqum/IoT analytics/Task 1/Wifi/validationData.csv")

############################################### Training dataset #################################################
### FEATURE EMGINEERING
## Changing value range
trainingData.iloc[:, 0:520] = np.where(trainingData.iloc[:, 0:520] <= 0,
        trainingData.iloc[:, 0:520] + 105,
        trainingData.iloc[:, 0:520] - 100)
trainingDatarep = trainingData.iloc[:, 0:520].replace(np.r_[1:16], 17)

trainingDatarep = trainingData.iloc[:, 0:520].replace(np.r_[81:200], 80)

trainingDatarep.iloc[:, 0:520] = np.where(trainingDatarep.iloc[:, 0:520] >= 0,
        trainingDatarep.iloc[:, 0:520] - 16,
        trainingDatarep.iloc[:, 0:520] - 0)

trainingDatarep = trainingDatarep.iloc[:, 0:520].replace(np.r_[-16:0],0)
other = trainingData.iloc[:, 520:529]
trainingDatarp  = pd.concat([trainingDatarep, other], axis = 1)
     
## Checking for duplicate rows
trainingDatarp = trainingDatarp.drop_duplicates(subset = None, keep='first', inplace=False)

## Checking duplicate columns
def getduplicateColumnNames(df):
    
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(trainingDatarp.shape[1]):
        # Select column at xth index.
        col = trainingData.iloc[:,x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, trainingDatarp.shape[1]):
            # Select column at yth index.
            otherCol = trainingDatarp.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)

getduplicateColumnNames(trainingData)

# Delete duplicate columns
trainingDatawoduplicates = trainingDatarp.drop(columns=getduplicateColumnNames(trainingDatarp))
 
#### Feature engineering
## Transforming and rotating 'Longitude' and 'Latitude' variables
angle=np.arctan(trainingDatawoduplicates['LATITUDE'][1]/trainingDatawoduplicates['LONGITUDE'][1])
#angle=89.91343355
angle=angle/math.pi
#alpha = 28.620334799694323
trainingDatawoduplicates['LONGITUDE']=-trainingDatawoduplicates['LONGITUDE']
A = trainingDatawoduplicates['LONGITUDE']*math.cos(angle) + trainingDatawoduplicates['LATITUDE']*math.sin(angle)
B = trainingDatawoduplicates['LATITUDE']*math.cos(angle) - trainingDatawoduplicates['LONGITUDE']*math.sin(angle)
plt.scatter(A,B)
trainingDatawoduplicates['LONGITUDE']=A
trainingDatawoduplicates['LATITUDE']=B
 
phone00 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 0]
phone01 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 1]
phone02 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 2]
phone03 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 3]
phone04 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 4]
phone05 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 5]
phone06 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 6]
phone07 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 7]
phone08 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 8]
phone09 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 9]
phone10 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 10]
phone11 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 11]
phone12 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 12]
phone13 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 13]
phone14 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 14]
phone15 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 15]
phone16 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 16]
phone17 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 17]
phone18 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 18]
phone19 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 19]
phone20 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 20]
phone21 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 21]
phone22 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 22]
phone23 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 23]
phone24 = trainingDatawoduplicates[trainingDatawoduplicates['PHONEID'] == 24]

from sklearn.preprocessing import MinMaxScaler
scaler =  MinMaxScaler()
def scalingfunction(df):
    scaler.fit(df)
    scaler.transform(df)

scalingfunction(phone00.iloc[:, 0:520])
scalingfunction(phone01.iloc[:, 0:520])
scalingfunction(phone02.iloc[:, 0:520])
scalingfunction(phone03.iloc[:, 0:520])
scalingfunction(phone04.iloc[:, 0:520])
scalingfunction(phone05.iloc[:, 0:520])
scalingfunction(phone06.iloc[:, 0:520])
scalingfunction(phone07.iloc[:, 0:520])
scalingfunction(phone08.iloc[:, 0:520])
scalingfunction(phone09.iloc[:, 0:520])
scalingfunction(phone10.iloc[:, 0:520])
scalingfunction(phone11.iloc[:, 0:520])
scalingfunction(phone12.iloc[:, 0:520])
scalingfunction(phone13.iloc[:, 0:520])
scalingfunction(phone14.iloc[:, 0:520])
scalingfunction(phone15.iloc[:, 0:520])
scalingfunction(phone16.iloc[:, 0:520])
scalingfunction(phone17.iloc[:, 0:520])
scalingfunction(phone18.iloc[:, 0:520])
scalingfunction(phone19.iloc[:, 0:520])
scalingfunction(phone20.iloc[:, 0:520])
scalingfunction(phone21.iloc[:, 0:520])
scalingfunction(phone22.iloc[:, 0:520])
scalingfunction(phone23.iloc[:, 0:520])
scalingfunction(phone24.iloc[:, 0:520])

tdtotal = pd.concat([phone00, phone01, phone02, phone03, phone04, phone05, phone06, phone07, phone08,
                     phone09, phone10, phone11, phone12, phone13, phone14, phone15, phone16,
                     phone17, phone18, phone19, phone20, phone21, phone22, phone23, phone24])
############################################### Validation dataset #################################################
### FEATURE EMGINEERING
## Changing value range
validationData.iloc[:, 0:520] = np.where(validationData.iloc[:, 0:520] <= 0,
        validationData.iloc[:, 0:520] + 105,
        validationData.iloc[:, 0:520] - 100)

validationDatarep = validationData.iloc[:, 0:520].replace(np.r_[1:16], 17)

validationDatarep = validationData.iloc[:, 0:520].replace(np.r_[81:200], 80)

validationDatarep.iloc[:, 0:520] = np.where(validationDatarep.iloc[:, 0:520] >= 0,
        validationDatarep.iloc[:, 0:520] - 16,
        validationDatarep.iloc[:, 0:520] - 0)

validationDatarep = validationDatarep.iloc[:, 0:520].replace(np.r_[-16:0],0)
otherv = validationData.iloc[:, 520:529]
validationDatarp  = pd.concat([validationDatarep, otherv], axis = 1)
     
## Checking for duplicate rows
validationDatarp = validationDatarp.drop_duplicates(subset = None, keep='first', inplace=False)
# Delete duplicate columns
validationwoduplicates = validationData.drop(columns=getduplicateColumnNames(trainingData))
## Transforming and rotating 'Longitude' and 'Latitude' variables

#alpha = 28.620334799694323
validationwoduplicates['LONGITUDE']=-validationwoduplicates['LONGITUDE']
Aval = validationwoduplicates['LONGITUDE']*math.cos(angle) + validationwoduplicates['LATITUDE']*math.sin(angle)
Bval = validationwoduplicates['LATITUDE']*math.cos(angle) - validationwoduplicates['LONGITUDE']*math.sin(angle)
plt.scatter(Aval,Bval)
validationwoduplicates['LONGITUDE']=Aval
validationwoduplicates['LATITUDE']=Bval


phone00v = validationwoduplicates[validationwoduplicates['PHONEID'] == 0]
phone01v = validationwoduplicates[validationwoduplicates['PHONEID'] == 1]
phone02v = validationwoduplicates[validationwoduplicates['PHONEID'] == 2]
phone03v = validationwoduplicates[validationwoduplicates['PHONEID'] == 3]
phone04v = validationwoduplicates[validationwoduplicates['PHONEID'] == 4]
phone05v = validationwoduplicates[validationwoduplicates['PHONEID'] == 5]
phone06v = validationwoduplicates[validationwoduplicates['PHONEID'] == 6]
phone07v = validationwoduplicates[validationwoduplicates['PHONEID'] == 7]
phone08v = validationwoduplicates[validationwoduplicates['PHONEID'] == 8]
phone09v = validationwoduplicates[validationwoduplicates['PHONEID'] == 9]
phone10v = validationwoduplicates[validationwoduplicates['PHONEID'] == 10]
phone11v = validationwoduplicates[validationwoduplicates['PHONEID'] == 11]
phone12v = validationwoduplicates[validationwoduplicates['PHONEID'] == 12]
phone13v = validationwoduplicates[validationwoduplicates['PHONEID'] == 13]
phone14v = validationwoduplicates[validationwoduplicates['PHONEID'] == 14]
phone15v = validationwoduplicates[validationwoduplicates['PHONEID'] == 15]
phone16v = validationwoduplicates[validationwoduplicates['PHONEID'] == 16]
phone17v = validationwoduplicates[validationwoduplicates['PHONEID'] == 17]
phone18v = validationwoduplicates[validationwoduplicates['PHONEID'] == 18]
phone19v = validationwoduplicates[validationwoduplicates['PHONEID'] == 19]
phone20v = validationwoduplicates[validationwoduplicates['PHONEID'] == 20]
phone21v = validationwoduplicates[validationwoduplicates['PHONEID'] == 21]
phone22v = validationwoduplicates[validationwoduplicates['PHONEID'] == 22]
phone23v = validationwoduplicates[validationwoduplicates['PHONEID'] == 23]
phone24v = validationwoduplicates[validationwoduplicates['PHONEID'] == 24]

from sklearn.preprocessing import MinMaxScaler
scaler =  MinMaxScaler()
def scalingfunction(df):
    scaler.fit(df)
    scaler.transform(df)

scalingfunction(phone00v.iloc[:, 0:520])
scalingfunction(phone01v.iloc[:, 0:520])
scalingfunction(phone02v.iloc[:, 0:520])
scalingfunction(phone03v.iloc[:, 0:520])
scalingfunction(phone04v.iloc[:, 0:520])
scalingfunction(phone05v.iloc[:, 0:520])
scalingfunction(phone06v.iloc[:, 0:520])
scalingfunction(phone07v.iloc[:, 0:520])
scalingfunction(phone08v.iloc[:, 0:520])
scalingfunction(phone09v.iloc[:, 0:520])
scalingfunction(phone10v.iloc[:, 0:520])
scalingfunction(phone11v.iloc[:, 0:520])
scalingfunction(phone12v.iloc[:, 0:520])
scalingfunction(phone13v.iloc[:, 0:520])
scalingfunction(phone14v.iloc[:, 0:520])
scalingfunction(phone15v.iloc[:, 0:520])
scalingfunction(phone16v.iloc[:, 0:520])
scalingfunction(phone17v.iloc[:, 0:520])
scalingfunction(phone18v.iloc[:, 0:520])
scalingfunction(phone19v.iloc[:, 0:520])
scalingfunction(phone20v.iloc[:, 0:520])
scalingfunction(phone21v.iloc[:, 0:520])
scalingfunction(phone22v.iloc[:, 0:520])
scalingfunction(phone23v.iloc[:, 0:520])
scalingfunction(phone24v.iloc[:, 0:520])

valtotal = pd.concat([phone00v, phone01v, phone02v, phone03v, phone04v, phone05v, phone06v, phone07v, phone08v,
                     phone09v, phone10v, phone11v, phone12v, phone13v, phone14v, phone15v, phone16v,
                     phone17v, phone18v, phone19v, phone20v, phone21v, phone22v, phone23v, phone24v])

######## MODEL FOR BUILDING #####
#Splitting data into train and test
x_train = trainingDatawoduplicates.iloc[:, 0:466]
y_train = trainingDatawoduplicates.iloc[: , 466:470]
x_test = validationwoduplicates.iloc[:, 0:466]
y_test = validationwoduplicates.iloc[: , 466:470]

RFbuilding = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_train, y_train['BUILDINGID'])
RFbuilding.score(x_test, y_test['BUILDINGID'])
# 10-Fold Cross validation
#RFCbuild_cv_score = cross_val_score(RFbuilding, x, y, cv=10)
RFCpred_build = RFbuilding.predict(x_test)
print(classification_report(y_test['BUILDINGID'], RFCpred_build))

#KNN
KNN_building = KNeighborsClassifier(n_neighbors=10).fit(x_train, y_train['BUILDINGID'])
KNN_building.score(x_test, y_test['BUILDINGID'])
#KNNbuild_cv_score = cross_val_score(KNN_building, x_test, y_test['BUILDINGID'], cv=10)
KNNpred_building = KNN_building.predict(x_test)
print(classification_report(y_test['BUILDINGID'], KNNpred_building))

#NN
NN_building = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(x_train, y_train['BUILDINGID'])
NN_building.score(x_test, y_test['BUILDINGID'])
#NNbuild_cv_score = cross_val_score(NN_building, x, y, cv=10)
NNpred_building = NN_building.predict(x_test)
print(classification_report(y_test['BUILDINGID'], NNpred_building))
 
#LR
LR_building = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train ,y_train['BUILDINGID'])
LR_building.score(x_test, y_test['BUILDINGID'])
#LRbuild_cv_score = cross_val_score(LR_building, x_test, y_test['BUILDINGID'], cv=10)
LRpred_building = LR_building.predict(x_test)
print(classification_report(y_test['BUILDINGID'], LRpred_building))
confusion_matrix(y_test['BUILDINGID'], LRpred_building)

#### PRINT ALL ACCURACIES AND KAPPAS
print("Accuracy:", accuracy_score(y_test['BUILDINGID'], LRpred_building))
print("Kappa:",cohen_kappa_score(y_test['BUILDINGID'], LRpred_building))

print("Accuracy:", accuracy_score(y_test['BUILDINGID'], NNpred_building))
print("Kappa:",cohen_kappa_score(y_test['BUILDINGID'], NNpred_building))

print("Accuracy:", accuracy_score(y_test['BUILDINGID'], KNNpred_building))
print("Kappa:",cohen_kappa_score(y_test['BUILDINGID'], KNNpred_building))

print("Accuracy:", accuracy_score(y_test['BUILDINGID'], RFCpred_build))
print("Kappa:",cohen_kappa_score(y_test['BUILDINGID'], RFCpred_build))

#### Reindexing df in order to use the predicted building as predictor
#Trainingset
#NNpredbuilddfytrain = pd.DataFrame(y_train_building)
#NNpredbuilddfytest = pd.DataFrame(y_test_building)
#NNpredbuilddftd = pd.concat([NNpredbuilddfytrain, NNpredbuilddfytest])
#NNpredbuilddftd = NNpredbuilddftd.rename(columns={'BUILDINGID': 'Buildpred'})
#trainingDatawoduplicates["Buildpred"] = NNpredbuilddftd
#Validationset
#NNpredbuilddfytrainval = pd.DataFrame(y_val_building)
#NNpredbuilddfytrainval = NNpredbuilddfytrainval.rename(columns={'BUILDINGID': 'Buildpred'})
#validationwoduplicates["Buildpred"] = NNpredbuilddftd
#### Reindexing df in order to use the predicted building as predictor
#columns_titles = [trainingDatawoduplicates.filter(like='WAP'), trainingData.filter(like='Buildpred'),trainingData.filter(like='LON'|'LAT'|'ID'|'REL'|'TIM')]
#trainingDatawoduplicates.isnull
#trainingDatawoduplicatespredbuild = trainingDatawoduplicates.copy()
#trainingDatawoduplicatespredbuildwaps = trainingDatawoduplicatespredbuild.filter(like="WAP")
#trainingDatawoduplicatespredbuildpb = trainingDatawoduplicatespredbuild.filter(like="Pred")
#trainingDatawoduplicatespredbuildoth = trainingDatawoduplicates.iloc[:, 467:476]
#trainingDatawoduplicatespredbuildv2 = [trainingDatawoduplicatespredbuildwaps, trainingDatawoduplicatespredbuildpb, trainingDatawoduplicatespredbuildoth]
#trainingDatawoduplicatespredbuildv3 = pd.concat([trainingDatawoduplicatespredbuildwaps, trainingDatawoduplicatespredbuildpb, trainingDatawoduplicatespredbuildoth], axis=1, sort=False)

######## MODEL FOR LONGITUDE #####
#Splitting data into train and test
tdtotal.columns.get_loc("BUILDINGID")
x_long = tdtotal.iloc[:, np.r_[0:466,469]]
y_long = tdtotal['LONGITUDE']
x_val_long = valtotal.iloc[:, np.r_[0:466,469]]
y_val_long = valtotal['LONGITUDE']

x_train['BUILDING'] = y_train['BUILDINGID']
x_test['BUILDING'] = LRpred_building

modelRF_long = RandomForestRegressor()
modelRF_long.fit(x_train,y_train['LONGITUDE'])
modelRF_long_pred = modelRF_long.predict(x_test)
round(modelRF_long.score(x_test,y_test['LONGITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LONGITUDE'], modelRF_long_pred)))
print("MAE:", mean_absolute_error(y_test['LONGITUDE'], modelRF_long_pred))
print("R2:",r2_score(y_test['LONGITUDE'], modelRF_long_pred))

modelKNN_long = KNeighborsRegressor(n_neighbors=10)
modelKNN_long.fit(x_train,y_train['LONGITUDE'])
modelKNN_long_pred = modelKNN_long.predict(x_test)
round(modelKNN_long.score(x_test,y_test['LONGITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LONGITUDE'], modelKNN_long_pred)))
print("MAE:", mean_absolute_error(y_test['LONGITUDE'], modelKNN_long_pred))
print("R2:",r2_score(y_test['LONGITUDE'], modelKNN_long_pred))

######## MODEL FOR LATITUDE #####
x_train['LONGITUDE'] = y_train['LONGITUDE']
x_test['LONGITUDE'] = modelKNN_long_pred

modelRF_lat = RandomForestRegressor()
modelRF_lat.fit(x_train,y_train['LATITUDE'])
modelRF_lat_pred = modelRF_lat.predict(x_test)
round(modelRF_lat.score(x_test,y_test['LATITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LATITUDE'], modelRF_lat_pred)))
print("MAE:", mean_absolute_error(y_test['LATITUDE'], modelRF_lat_pred))
print("R2:",r2_score(y_test['LATITUDE'], modelRF_lat_pred))

modelKNN_lat = KNeighborsRegressor(n_neighbors=10)
modelKNN_lat.fit(x_train,y_train['LATITUDE'])
modelKNN_lat_pred = modelKNN_lat.predict(x_test)
round(modelKNN_lat.score(x_test,y_test['LATITUDE']), 10)

print("RMSE:", sqrt(mean_squared_error(y_test['LATITUDE'], modelKNN_lat_pred)))
print("MAE:", mean_absolute_error(y_test['LATITUDE'], modelKNN_lat_pred))
print("R2:",r2_score(y_test['LATITUDE'], modelKNN_lat_pred))


######## MODEL FOR FLOOR #####
#Splitting data into train and test
trainingDatawoduplicates.columns.get_loc("LATITUDE")
x_floor = trainingDatawoduplicates.iloc[:, np.r_[0:468,469]]
y_floor = trainingDatawoduplicates['FLOOR']
x_val_floor = validationwoduplicates.iloc[:, np.r_[0:468,469]]
y_val_floor = validationwoduplicates['FLOOR']

## PERFORMING REGRESSION MODELS
KNN_floor = KNeighborsClassifier(n_neighbors=10)
KNN_floor.fit(x_floor, y_floor)
KNNpredcitions_floor = KNN_floor.predict(x_val_floor)
round(KNN_floor.score(x_val_floor, y_val_floor), 10)
#ACCURACY AND KAPPA  
#RF
print("Accuracy:", accuracy_score(y_val_floor, KNNpredcitions_floor))
print("Kappa:",cohen_kappa_score(y_val_floor, KNNpredcitions_floor))


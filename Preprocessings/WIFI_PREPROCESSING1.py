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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

#model metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
#cross validation
from sklearn.model_selection import train_test_split
trainingData = pd.read_csv("C:/Users/andre/Desktop/Ubiqum/IoT analytics/Task 1/Wifi/trainingData.csv")
validationData = pd.read_csv("C:/Users/andre/Desktop/Ubiqum/IoT analytics/Task 1/Wifi/validationData.csv")

# =============================================================================
# Training dataset preprocessing
# =============================================================================

# =============================================================================
# FEATURE EMGINEERING
# =============================================================================

## Changing value range

trainingData.iloc[:, 0:520] = np.where(trainingData.iloc[:, 0:520] <= 0,
        trainingData.iloc[:, 0:520] + 105,
        trainingData.iloc[:, 0:520] - 100)
  
trainingDatarep = trainingData.iloc[:, 0:520].replace(np.r_[1:16], 17)
trainingDatarep = trainingData.iloc[:, 0:520].replace(np.r_[71:200], 70)

trainingDatarep.iloc[:, 0:520] = np.where(trainingDatarep.iloc[:, 0:520] >= 0,
        trainingDatarep.iloc[:, 0:520] - 16,
        trainingDatarep.iloc[:, 0:520] - 0)

trainingDatarep = trainingDatarep.iloc[:, 0:520].replace(np.r_[-16:0],0)
other = trainingData.iloc[:, 520:529]
trainingDatarp  = pd.concat([trainingDatarep, other], axis = 1)
     
# Delete duplicate rows

trainingDatarp = trainingDatarp.drop_duplicates(subset = None, keep='first', inplace=False)

## Checking duplicate columns

def getduplicateColumnNames(df):
    
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(trainingData.shape[1]):
        # Select column at xth index.
        col = trainingData.iloc[:,x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, trainingData.shape[1]):
            # Select column at yth index.
            otherCol = trainingData.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)

getduplicateColumnNames(trainingData)

# Delete duplicate columns

trainingDatawoduplicates = trainingDatarp.drop(columns=getduplicateColumnNames(trainingDatarp))

# NORMALIZER TRANSFORMATION AND MIN MAX
#trainingDatawoduplicates.columns.get_loc('LATITUDE')

#trainingDatawoduplicates.iloc[:,467]

#transformer = Normalizer().fit_transform(trainingDatawoduplicates.iloc[:, 0:466])
#traintrans = pd.DataFrame(transformer, columns=trainingDatawoduplicates.iloc[:, 0:466].columns)
#old = pd.DataFrame(trainingData.iloc[:, 520:529])
#traintransf = pd.concat([traintrans, old], axis = 1, ignore_index = True, join="inner")
#traintransf.columns = trainingDatawoduplicates.columns

#Check NA's
#traintransf.isna().sum().sum()
#old.isna().sum().sum()

## Checking for duplicate rows
#trainingData = trainingData.drop_duplicates(subset = None, keep='first', inplace=False)

#LONGITUDE = wifi["LONGITUDE"]*np.cos(angle) + wifi["LATITUDE"]*np.sin(angle)
#LATITUDE = wifi["LATITUDE"]*np.cos(angle) - wifi["LONGITUDE"]*np.sin(angle)
#wifi["LONGITUDE"] = LONGITUDE
#wifi["LATITUDE"] = LATITUDE
#vlong = validate["LONGITUDE"]*np.cos(angle) +validate["LATITUDE"]*np.sin(angle)
#vlat = validate["LATITUDE"]*np.cos(angle) - validate["LONGITUDE"]*np.sin(angle)
#validate["LONGITUDE"] = vlong
#validate["LATITUDE"] = vlat
#### Feature engineering
## Transforming and rotating 'Longitude' and 'Latitude' variables
angle=np.arctan(trainingDatawoduplicates['LATITUDE'][0]/trainingDatawoduplicates['LONGITUDE'][0])
angle=angle/math.pi
#trainingDatawoduplicates['LONGITUDE']=-trainingDatawoduplicates['LONGITUDE']
LONGITUDE = trainingDatawoduplicates['LONGITUDE']*np.cos(angle) + trainingDatawoduplicates['LATITUDE']*np.sin(angle)
LATITUDE = trainingDatawoduplicates['LATITUDE']*np.cos(angle) - trainingDatawoduplicates['LONGITUDE']*np.sin(angle)
plt.scatter(LONGITUDE,LATITUDE)
trainingDatawoduplicates['LONGITUDE']=LONGITUDE
trainingDatawoduplicates['LATITUDE']=LATITUDE

#trainingDatawoduplicates['HigherWAP'] = trainingDatawoduplicates.iloc[:, 0:520].max(axis=1)

# =============================================================================
# Validation dataset
# =============================================================================

# =============================================================================
# FEATURE EMGINEERING
# =============================================================================

## Changing value range
validationData.iloc[:, 0:520] = np.where(validationData.iloc[:, 0:520] <= 0,
        validationData.iloc[:, 0:520] + 105,
        validationData.iloc[:, 0:520] - 100)

validationDatarep = validationData.iloc[:, 0:520].replace(np.r_[1:16], 17)
validationDatarep = validationData.iloc[:, 0:520].replace(np.r_[81:200],80)

validationDatarep.iloc[:, 0:520] = np.where(validationDatarep.iloc[:, 0:520] >= 0,
        validationDatarep.iloc[:, 0:520] - 16,
        validationDatarep.iloc[:, 0:520] - 0)

validationDatarep = validationDatarep.iloc[:, 0:520].replace(np.r_[-16:0],0)

# Delete duplicate columns

validationwoduplicates = validationData.drop(columns=getduplicateColumnNames(trainingDatarp))

##### NORMALIZER AND MIN MAX TRANSFORMATION

#transformerv = Normalizer().fit_transform(validationwoduplicates.iloc[:, 0:466])
#valtrans = pd.DataFrame(transformerv, columns=validationwoduplicates.iloc[:, 0:466].columns)
#oldval = pd.DataFrame(validationData.iloc[:,520:529])
#valtransf = pd.concat([valtrans, oldval], axis = 1)
#valtransf.columns = trainingDatawoduplicates.columns
 
#Check NA
#traintransf.isna().sum().sum()
#valfinal.isna().sum().sum()
## Transforming and rotating 'Longitude' and 'Latitude' variables
#plt.scatter(validationData["LONGITUDE"],validationData["LATITUDE"])
#alpha = 28.620334799694323
#validationwoduplicates['LONGITUDE']=-validationwoduplicates['LONGITUDE']
Longval = validationwoduplicates['LONGITUDE']*np.cos(angle) + validationwoduplicates['LATITUDE']*np.sin(angle)
Latival = validationwoduplicates['LATITUDE']*np.cos(angle) - validationwoduplicates['LONGITUDE']*np.sin(angle)
plt.scatter(Aval,Bval)
validationwoduplicates['LONGITUDE']=Longval
validationwoduplicates['LATITUDE']=Latival
#validationwoduplicates['HigherWAP'] = validationwoduplicates.iloc[:, 0:520].max(axis=1) 


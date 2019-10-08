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
validationData = pd.read_csv("C:/Users/andre/Desktop/Ubiqum/IoT analytics/Task 1/Wifi/validationData.csv")
trainingData = pd.read_csv("C:/Users/andre/Desktop/Ubiqum/IoT analytics/Task 1/Wifi/trainingData.csv")

# =============================================================================
# Training dataset preprocessing
# =============================================================================

# =============================================================================
# FEATURE EMGINEERING
# =============================================================================
#trainingData = pd.concat([trainingData, validationData], axis=0)
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
     
trainingDatarp = trainingDatarp[trainingDatarp['PHONEID'] != 17]
trainingDatarp = trainingDatarp[trainingDatarp['PHONEID'] != 11]

transformer = Normalizer().fit_transform(trainingData.iloc[:, 0:520])
traintrans = pd.DataFrame(transformer)
old = pd.DataFrame(trainingData.iloc[:,520:529])
traintransf = pd.concat([traintrans, other], axis = 1)
traintransf.columns = trainingData.columns

# Delete duplicate rows

traintransf   = traintransf.drop_duplicates(subset = None, keep='first', inplace=False)

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

# Delete duplicate columns

trainingDatawoduplicates = traintransf.drop(columns=getduplicateColumnNames(traintransf))

# =============================================================================
# FEATURE EMGINEERING
# =============================================================================

## Changing value range
 
#trainingData = pd.concat([trainingData, validationData], axis=0)
validationData.iloc[:, 0:520] = np.where(validationData.iloc[:, 0:520] <= 0,
        validationData.iloc[:, 0:520] + 105,
        validationData.iloc[:, 0:520] - 100)
  
validationDatarep = validationData.iloc[:, 0:520].replace(np.r_[1:16], 17)
validationDatarep = validationData.iloc[:, 0:520].replace(np.r_[71:200], 70)

validationDatarep.iloc[:, 0:520] = np.where(validationDatarep.iloc[:, 0:520] >= 0,
        validationDatarep.iloc[:, 0:520] - 16,
        validationDatarep.iloc[:, 0:520] - 0)

validationDatarep = validationDatarep.iloc[:, 0:520].replace(np.r_[-16:0],0)
otherv = validationData.iloc[:, 520:529]

validationDatarp  = pd.concat([validationDatarep, otherv], axis = 1)
     
transformerv = Normalizer().fit_transform(validationDatarp.iloc[:, 0:520])
traintransv = pd.DataFrame(transformerv)
oldv = pd.DataFrame(validationData.iloc[:,520:529])
traintransfv = pd.concat([traintransv, oldv], axis = 1)
traintransfv.columns = validationData.columns

# Delete duplicate columns

validationDatawoduplicates = traintransfv.drop(columns=getduplicateColumnNames(trainingDatarp))

mergedDataset = pd.concat([trainingDatawoduplicates, validationDatawoduplicates], axis=0)

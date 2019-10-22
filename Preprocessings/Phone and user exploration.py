import statistics
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import math
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
#model metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
#cross validation
from sklearn.model_selection import train_test_split

trainingData = pd.read_csv("C:/Users/andre/Desktop/Ubiqum/IoT analytics/Task 1/Wifi/trainingData.csv")
validationData = pd.read_csv("C:/Users/andre/Desktop/Ubiqum/IoT analytics/Task 1/Wifi/validationData.csv")

# =============================================================================
# Training data analysis
# =============================================================================
# First of all what I want to look is the floor distribution's for each dataset
floor0 = trainingData[trainingData['FLOOR'] == 0]
floor1 = trainingData[trainingData['FLOOR'] == 1]
floor2 = trainingData[trainingData['FLOOR'] == 2]
floor3 = trainingData[trainingData['FLOOR'] == 3]
floor4 = trainingData[trainingData['FLOOR'] == 4]
floor5 = trainingData[trainingData['FLOOR'] == 5]

floor0v = validationData[validationData['FLOOR'] == 0]
floor1v = validationData[validationData['FLOOR'] == 1]
floor2v = validationData[validationData['FLOOR'] == 2]
floor3v = validationData[validationData['FLOOR'] == 3]
floor4v = validationData[validationData['FLOOR'] == 4]
floor5v = validationData[validationData['FLOOR'] == 5]

# Here I change the value range of the training set in order to change the scale to positive 
# and to apply the preprocessing that changes the scale
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

validationData.iloc[:, 0:520] = np.where(validationData.iloc[:, 0:520] <= 0,
        validationData.iloc[:, 0:520] + 105,
        validationData.iloc[:, 0:520] - 100)

validationDatarep = validationData.iloc[:, 0:520].replace(np.r_[1:16], 17)
validationDatarep = validationData.iloc[:, 0:520].replace(np.r_[81:200],80)

validationDatarep.iloc[:, 0:520] = np.where(validationDatarep.iloc[:, 0:520] >= 0,
        validationDatarep.iloc[:, 0:520] - 16,
        validationDatarep.iloc[:, 0:520] - 0)
validationDatarep = validationDatarep.iloc[:, 0:520].replace(np.r_[-16:0],0)

validationDatarp  = pd.concat([validationDatarep, other], axis = 1)

## Here I created the histogram of the WAP signal distribution
WAPs = validationDatarp.filter(like="WAP")
WAPsmelted = pd.melt(WAPs)
plt.hist(WAPsmelted['value'])
# As the plot doesn't look good because the 0 values are much more represented,
# here I create a plot without those 0 values
WAPSmeltedw0 = WAPsmelted.loc[(WAPsmelted.iloc[:,1] != 0)]
plt.hist(WAPSmeltedw0['value'], bins =21)

# =============================================================================
# HARDCODE :(
# =============================================================================

# The idea here was to check the distributions of the phones

phone00 = trainingData[trainingData['PHONEID'] == 0]
phone01 = trainingData[trainingData['PHONEID'] == 1]
phone02 = trainingData[trainingData['PHONEID'] == 2]
phone03 = trainingData[trainingData['PHONEID'] == 3]
phone04 = trainingData[trainingData['PHONEID'] == 4]
phone05 = trainingData[trainingData['PHONEID'] == 5]
phone06 = trainingData[trainingData['PHONEID'] == 6]
phone07 = trainingData[trainingData['PHONEID'] == 7]
phone08 = trainingData[trainingData['PHONEID'] == 8]
phone09 = trainingData[trainingData['PHONEID'] == 9]
phone10 = trainingData[trainingData['PHONEID'] == 10]
phone11 = trainingData[trainingData['PHONEID'] == 11]
phone12 = trainingData[trainingData['PHONEID'] == 12]
phone13 = trainingData[trainingData['PHONEID'] == 13]
phone14 = trainingData[trainingData['PHONEID'] == 14]
phone15 = trainingData[trainingData['PHONEID'] == 15]
phone16 = trainingData[trainingData['PHONEID'] == 16]
phone17 = trainingData[trainingData['PHONEID'] == 17]
phone18 = trainingData[trainingData['PHONEID'] == 18]
phone19 = trainingData[trainingData['PHONEID'] == 19]
phone20 = trainingData[trainingData['PHONEID'] == 20]
phone21 = trainingData[trainingData['PHONEID'] == 21]
phone22 = trainingData[trainingData['PHONEID'] == 22]
phone23 = trainingData[trainingData['PHONEID'] == 23]
phone24 = trainingData[trainingData['PHONEID'] == 24]

# This function I created melts the values of the WAPs by phone
def melting(df):
    df = df.filter(like="WAP")
    df = pd.melt(df)
    df = df.loc[(df.iloc[:,1] != 0)]
    return df

# Here I apply the function to all the dataframes
phone00m = melting(phone00)
phone01m = melting(phone01) 
phone02m = melting(phone02)
phone03m = melting(phone03) 
phone04m = melting(phone04)
phone05m = melting(phone05) 
phone06m = melting(phone06)
phone07m = melting(phone07)
phone08m = melting(phone08)
phone09m = melting(phone09) 
phone10m = melting(phone10)
phone11m = melting(phone11)
phone12m = melting(phone12)
phone13m = melting(phone13) 
phone14m = melting(phone14)
phone15m = melting(phone15)
phone16m = melting(phone16)
phone17m = melting(phone17) 
phone18m = melting(phone18)
phone19m = melting(phone19)
phone20m = melting(phone20)
phone21m = melting(phone21)
phone22m = melting(phone22) 
phone23m = melting(phone23)
phone24m = melting(phone24)

# =============================================================================
# HISTOGRAMS
# =============================================================================

plt.hist(phone01m['value'], bins =21)
plt.hist(phone03m['value'], bins =21)
plt.hist(phone06m['value'], bins =21)
plt.hist(phone07m['value'], bins =50)
plt.hist(phone08m['value'], bins =21)
plt.hist(phone10m['value'], bins =21)
plt.hist(phone11m['value'], bins =50)
plt.hist(phone13m['value'], bins =21)
plt.hist(phone14m['value'], bins =21)
plt.hist(phone16m['value'], bins =21)
plt.hist(phone17m['value'], bins =50)
plt.hist(phone18m['value'], bins =21)
plt.hist(phone19m['value'], bins =50)
plt.hist(phone22m['value'], bins =21)
plt.hist(phone23m['value'], bins =21)
plt.hist(phone24m['value'], bins =21)

x = trainingData.groupby('PHONEID')
x.first()

# =============================================================================
# VALIDATION ANALYSIS
# =============================================================================

# Here I repeat the same process for the validation set
# =============================================================================
# HARDCODE :(
# =============================================================================
phone00v = validationData[validationData['PHONEID'] == 0]
phone01v = validationData[validationData['PHONEID'] == 1]
phone02v = validationData[validationData['PHONEID'] == 2]
phone03v = validationData[validationData['PHONEID'] == 3]
phone04v = validationData[validationData['PHONEID'] == 4]
phone05v = validationData[validationData['PHONEID'] == 5]
phone06v = validationData[validationData['PHONEID'] == 6]
phone07v = validationData[validationData['PHONEID'] == 7]
phone08v = validationData[validationData['PHONEID'] == 8]
phone09v = validationData[validationData['PHONEID'] == 9]
phone10v = validationData[validationData['PHONEID'] == 10]
phone11v = validationData[validationData['PHONEID'] == 11]
phone12v = validationData[validationData['PHONEID'] == 12]
phone13v = validationData[validationData['PHONEID'] == 13]
phone14v = validationData[validationData['PHONEID'] == 14]
phone15v = validationData[validationData['PHONEID'] == 15]
phone16v = validationData[validationData['PHONEID'] == 16]
phone17v = validationData[validationData['PHONEID'] == 17]
phone18v = validationData[validationData['PHONEID'] == 18]
phone19v = validationData[validationData['PHONEID'] == 19]
phone20v = validationData[validationData['PHONEID'] == 20]
phone21v = validationData[validationData['PHONEID'] == 21]
phone22v = validationData[validationData['PHONEID'] == 22]
phone23v = validationData[validationData['PHONEID'] == 23]
phone24v = validationData[validationData['PHONEID'] == 24]

def melting(df):
    df = df.filter(like="WAP")
    df = pd.melt(df)
    df = df.loc[(df.iloc[:,1] != 0)]
    return df


phone00mv = melting(phone00v)
phone01mv = melting(phone01v) 
phone02mv = melting(phone02v)
phone03mv = melting(phone03v) 
phone04mv = melting(phone04v)
phone05mv = melting(phone05v) 
phone06mv = melting(phone06v)
phone07mv = melting(phone07v)
phone08mv = melting(phone08v)
phone09mv = melting(phone09v) 
phone10mv = melting(phone10v)
phone11mv = melting(phone11v)
phone12mv = melting(phone12v)
phone13mv = melting(phone13v) 
phone14mv = melting(phone14v)
phone15mv = melting(phone15v)
phone16mv = melting(phone16v)
phone17mv = melting(phone17v) 
phone18mv = melting(phone18v)
phone19mv = melting(phone19v)
phone20mv = melting(phone20v)
phone21mv = melting(phone21v)
phone22mv = melting(phone22v) 
phone23mv = melting(phone23v)
phone24mv = melting(phone24v)

# =============================================================================
# HISTOGRAMS
# =============================================================================

plt.hist(phone00mv['value'], bins =21)
plt.hist(phone01mv['value'], bins =21)
plt.hist(phone03mv['value'], bins =21)
plt.hist(phone05mv['value'], bins =21)
plt.hist(phone06mv['value'], bins =21)
plt.hist(phone07mv['value'], bins =21)
plt.hist(phone08mv['value'], bins =21)
plt.hist(phone09mv['value'], bins =21)
plt.hist(phone10mv['value'], bins =21)
plt.hist(phone11mv['value'], bins =21)
plt.hist(phone12mv['value'], bins =21)
plt.hist(phone13mv['value'], bins =21)
plt.hist(phone14mv['value'], bins =21)
plt.hist(phone15mv['value'], bins =21)
plt.hist(phone20mv['value'], bins =21)
plt.hist(phone21mv['value'], bins =21)



# =============================================================================
# USERID TD Analysis
# =============================================================================

# Here I repeat the same process for userID on the training set

user01 = trainingData[trainingData['USERID'] == 1]
user02 = trainingData[trainingData['USERID'] == 2]
user03 = trainingData[trainingData['USERID'] == 3]
user04 = trainingData[trainingData['USERID'] == 4]
user05 = trainingData[trainingData['USERID'] == 5]
user06 = trainingData[trainingData['USERID'] == 6]
user07 = trainingData[trainingData['USERID'] == 7]
user08 = trainingData[trainingData['USERID'] == 8]
user09 = trainingData[trainingData['USERID'] == 9]
user10 = trainingData[trainingData['USERID'] == 10]
user11 = trainingData[trainingData['USERID'] == 11]
user12 = trainingData[trainingData['USERID'] == 12]
user13 = trainingData[trainingData['USERID'] == 13]
user14 = trainingData[trainingData['USERID'] == 14]
user15 = trainingData[trainingData['USERID'] == 15]
user16 = trainingData[trainingData['USERID'] == 16]
user17 = trainingData[trainingData['USERID'] == 17]
user18 = trainingData[trainingData['USERID'] == 18]

user01 = melting(user01)
user02 = melting(user02)
user03 = melting(user03)
user04 = melting(user04)
user05 = melting(user05)
user06 = melting(user06)
user07 = melting(user07)
user08 = melting(user08)
user09 = melting(user09)
user10 = melting(user10)
user11 = melting(user11)
user12 = melting(user12)
user13 = melting(user13)
user14 = melting(user14)
user15 = melting(user15)
user16 = melting(user16)
user17 = melting(user17)
user18 = melting(user18)

plt.hist(user01['value'], bins =21)
plt.hist(user02['value'], bins =21)
plt.hist(user03['value'], bins =21)
plt.hist(user05['value'], bins =21)
plt.hist(user06['value'], bins =21)
plt.hist(user07['value'], bins =21)
plt.hist(user08['value'], bins =21)
plt.hist(user09['value'], bins =21)
plt.hist(user10['value'], bins =21)
plt.hist(user11['value'], bins =21)
plt.hist(user12['value'], bins =21)
plt.hist(user13['value'], bins =21)
plt.hist(user14['value'], bins =21)
plt.hist(user15['value'], bins =21)
plt.hist(user16['value'], bins =21)
plt.hist(user17['value'], bins =21)
plt.hist(user18['value'], bins =21)


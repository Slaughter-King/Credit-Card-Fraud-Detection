#             IMPORTING THE DEPENDENCIES

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

#              DATA COLLECTION AND ANALYSIS

# loading the data from csvfile to a panda Dataframe
parkinsons_data = pd.read_csv('D:\khank\Programming Languages\ML file\parkinsons.csv')
# Printing the first five rows of the data frame
parkinsons_data.head()

# Number of rows and colums in frame
parkinsons_data.shape

# Getting more information about the data set
parkinsons_data.info()

# Checking for missing values in each column
parkinsons_data.isnull().sum()

# Getting some statistical measures about the data
parkinsons_data.describe()

# Distribution of target value
parkinsons_data['status'].value_counts() 
# 1-->parkisnson's positive
# 0-->healthy

#               DATA PRE-PROCESSING

# Separating the features and target
X=parkinsons_data.drop(columns=['name','status'],axis=1)
Y=parkinsons_data['status']
print(X)
print(Y)
# splitting the data to test data and training data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

#               DATA STANDARDISATION
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#                     MODEL TRAINING
# Support vector machine model
model=svm.SVC(kernel='linear')
# Training the SVM model with training data
model.fit(X_train,Y_train)

#             MODEL EVALUATION
# Accuracy score on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print('Accuracy score of training data',training_data_accuracy)

# Accuracy score on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy score of test data',test_data_accuracy)

#            BUILDING A PRERDICTIVE SYSTEM
input_data=(198.38300,215.20300,193.10400,0.00212,0.00001,0.00113,0.00135,0.00339,0.01263,0.11100,0.00640,0.00825,0.00951,0.01919,0.00119,30.77500,0.465946,0.738703,-7.067931,0.175181,1.512275,0.096320)
# changing the data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

# reshape the numpy array
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

# Standardize the data
std_data=scaler.transform(input_data_reshaped)
prediction=model.predict(std_data)
print(prediction)

if(prediction[0]==1):
    print('Person has Parkinson')
else:
    print('Person does not have Parkinson')    
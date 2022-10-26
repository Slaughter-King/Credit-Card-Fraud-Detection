#           IMPORTING DEPENDENCIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the data set to a pandas DataFrame
credit_card_data = pd.read_csv('D:\khank\Programming Languages\ML file\creditcard.csv\creditcard.csv')
credit_card_data.head() # Gives first five rows of data set
credit_card_data.tail() # Gives last five rows of data set

# Data Set information
credit_card_data.info()

# Checking the number of missing values in each column
credit_card_data.isnull()

# Distribution of legit transactions and fraudulent transactions
credit_card_data['Class'].value_counts() #This data set is highly unbalanced
# 0-->Normal Transactions ; 1-->Fraudulent transactions

# Searating the data for analysis
legit = credit_card_data[credit_card_data.Class==0]
fraud = credit_card_data[credit_card_data.Class==1]
print(legit.shape)
print(fraud.shape)

# Statistical measures of the data
legit.Amount.describe()
fraud.Amount.describe()

# Compare the values for both transactions
credit_card_data.groupby('Class').mean()

# Undersampling
# Build a sample dataset containing similar distribution of normal transactions and fraudulent transactions
# Number of fraudulent transactions-->492
legit_sample = legit.sample(n=492)

# Contactenating two DataFrames
new_dataset = pd.concat([legit_sample, fraud], axis=0)
# axis 0 means rows and axis 1 means columns
new_dataset.head()
new_dataset.tail( )
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()
# splitting the data into features and targets
X = new_dataset.drop(columns='Class',axis =1)
Y= new_dataset['Class']
print(X)
# Splitting the data into training data and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
# Model Training
model=LogisticRegression()
# Training the logistic regression model with training data
model.fit(X_train,Y_train)
# Model Evaluation
# Accuracy Score
# Accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy) # Accuracy on training dat
# Accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(test_data_accuracy) # Accuracy score on test data


import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Import all the necessary libraries


# Use the UCI ML Parkinsons dataset


df=pd.read_csv('parkinsons.data')
df.head()

print(labels[labels==1].shape[0], labels[labels==0].shape[0])

# Normalize the feature values to lie beween -1 and 1.

scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

#Use 20 percent of the data for test

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

#Use the XGBClassifier function to train the model

model=XGBClassifier()
model.fit(x_train,y_train)

#Predict the values of the test set with the XGB Classifier

y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

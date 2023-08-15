# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:52:55 2023

@author: Admin
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sna
%matplotlib inline 
import warnings 
warnings.filterwarnings('ignore')
current_directory=os.getcwd()
current_directory
df=pd.read_csv(r'C:\Users\Admin\Downloads\19th,20th\adult.csv')
df
df.shape
df.head()
df.info()
#Encode ?as NANs
df[df=='?']=np.nan
df.info()
#Impute missing values with mode
for col in['workclass','occupation','native.country']:
    df[col].fillna(df[col].mode()[0],inplace=True)
 #check again for missing values
df.isnull().sum()   
#Setting feature vector and target variable
x=df.drop(['income'],axis=1)    
y=df['income']
x.head()
#Split data into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
#Feature Engineering
#Encode categorical variables
from sklearn import preprocessing
categorical=['workclass','education','marital.status','occupation','relationship','race','sex','native.country']
for feature in categorical:
    le=preprocessing.LabelEncoder()
    x_train[feature]=le.fit_transform(x_train[feature])
    x_test[feature]=le.transform(x_test[feature])
#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x.columns)    
x_test=pd.DataFrame(scaler.transform(x_test),columns=x.columns)
x_train.head()
#Logistic Regressionmodel with all features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
print('Logistic regression accuracy score with all the features:{0:0.4f}'.format(accuracy_score(y_test,y_pred)))
#Logistic Regression with PCA
from sklearn.decomposition import PCA
pca=PCA()
x_train=pca.fit_transform(x_train)
pca.explained_variance_ratio_
#LogisticRegression with first 13 features
x=df.drop(['income','native.country'],axis=1)
y=df['income']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
categorical=['workclass','education','marital.status','occupation','relationship','race','sex']
for feature in categorical:
    le=preprocessing.LabelEncoder()
    x_train[feature]=le.fit_transform(x_train[feature])
    x_test[feature]=le.transform(x_test[feature])
x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x.columns)
x_test=pd.DataFrame(scaler.transform(x_test),columns=x.columns)
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
print('Logistic Regression accuracy score with the first 13 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

x=df.drop(['income'],axis=1)
y=df['income']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
categorical=['workclass','education','marital.status','occupation','relationship','race','sex','native.country']
for feature in categorical:
    le=preprocessing.LabelEncoder()
    x_train[feature]=le.fit_transform(x_train[feature])
    x_test[feature]=le.transform(x_test[feature])
x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x.columns)    
pca=PCA()
pca.fit(x_train)
cumsum=np.cumsum(pca.explained_variance_ratio_)
dim=np.argmax(cumsum>=0.90)+1
print('The number of dimensions required to preserve 90% of variance is',dim)
#PLot explained variance ratio with number of dimensions
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,14,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

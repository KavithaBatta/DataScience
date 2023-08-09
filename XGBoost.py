# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 20:54:59 2023

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv(r'C:\Users\Admin\Desktop\NIT Data Science Course\3rd july\3rd\7.XGBOOST\Churn_Modelling.csv')
dataset
x=dataset.iloc[:,3:-1].values
x
y=dataset.iloc[:,-1].values
y
print(x)
print(y)
#Encoding categorical data
#LAbel encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
print(x)
#One Hotencoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)
#splitting the dataset into Training and Testing set
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#Training XGBOOSt on the Training set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(x_train, y_train)
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.00001, max_delta_step=0, max_depth=10,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=400, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None) 
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

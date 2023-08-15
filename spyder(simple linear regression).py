# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 13:36:28 2023

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r'C:\Users\Admin\Downloads\7th\Salary_Data.csv')
dataset
x=dataset.iloc[:,:-1]
x
y=dataset.iloc[:,1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()
m=regressor.coef_
m
c=regressor.intercept_
y_=mx+c
y_=9312*12+26780
y_
bias=regressor.score(x_train,y_train)
bias
variance=regressor.score(x_test,y_test)
variance

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:38:50 2023

@author: Admin
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
dataset=pd.read_csv(r'C:\Users\Admin\Downloads\1.POLYNOMIAL REGRESSION\emp_sal.csv')
dataset
x=dataset.iloc[:,1:2].values
x
y=dataset.iloc[:,2]
y
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)
plt.scatter(x,y, color='red')
plt.plot(x,lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[10]]))

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 22:35:39 2023

@author: Admin
"""

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Import the dataset
dataset=pd.read_csv(r'C:\Users\Admin\Desktop\NIT Data Science Course\23rd, 26th jun\23rd, 26th\Social_Network_Ads.csv')
dataset
x=dataset.iloc[:,[2,3]].values
x
y=dataset.iloc[:,-1]
y
#Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#Feature scaling
'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
'''
#Training the Naive bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
#Predicting the test set results
y_pred=classifier.predict(x_test)
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
print(cm)
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac
print(ac)
bias=classifier.score(x_train,y_train)
bias
#Visualizing the trainimg set results
from matplotlib.colors import ListedColormap
x_test,y_test=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_test[:,0].min()-1,stop=x_test[:,0].max()+1,step=0.01),
                  np.arange(start=x_test[:,1].min()-1,stop=x_test[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                                      alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())    
plt.ylim(x2.min(),x2.max())
for i,j in  enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test==j,0],x_test[y_test==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('Naive Bayes(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
#Visulalizing the test results
from matplotlib.colors import ListedColormap
x_test,y_test=x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_test[:,0].min()-1,stop=x_test[:,0].max()+1,step=0.01),
                  np.arange(start=x_test[:,1].min()-1,stop=x_test[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                                      alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x1.min(),x2.max())
for i,j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test==j,0],x_test[y_test==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('Naive Bayes(Test set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

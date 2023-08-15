# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:02:01 2023

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#Data preparation
Load data
df=pd.read_csv(r'C:\Users\Admin\Downloads\16th\avocado.csv')
df
df.info()
df.head()
Missing value checking
df.isnull().sum()
Dropping unnecessary columns
df = df.drop(['Unnamed: 0','4046','4225','4770','Date'],axis=1)
df.head()
def get_avarage(df,column):
    return sum(df[column])/len(df)
    def get_avarage_between_two_columns(df,column1,column2):
        List=list(df[column1].unique())
        average=[]
        for i in List :
            x=df[df[column]==i]
            column1_average=get_avarage(x, column2)
            average.append(column1_average)
            df_column1_column2=pd.DataFrame({'column1':List,'column2':average})
            column1_column2_sorted_index=df_column1_column2.column2.sort_values(ascending=False).index.values
            column1_column2_sorted_data=df_column1_column2,reindex(column1_column2_sorted_index)
            return column1_column2_sorted_data
        def plot(data,xlabel,ylabel):
            plt.figure(figsize=(15,5))
            ax=sns.barplot(x=data.column1,y=data.column2,palette='rocket')
            plt.xticks(rotation=90)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(('Avarage'+ylabel+' of Avacado according to '+xlabel))
     #Lowest and highest prices of Avocado
    data1 = get_avarage_between_two_columns(df,'region','AveragePrice')
    plot(data1,'region','Price ($)')
    

            
            
            
        

   
        
  
    
   

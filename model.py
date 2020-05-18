# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:52:53 2020

@author: NAGAMANIKANTA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('dataset.csv')

df.drop(columns='index',inplace=True)

df['having_Sub_Domain'].replace(-1,0,inplace=True)

df.replace(-1,0,inplace=True)

X=df.iloc[:,:-1].values

y=df.iloc[:,30:31].values

from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder(categories='auto')
X=oh.fit_transform(X).toarray() 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model.logistic import LogisticRegression

cls =LogisticRegression(random_state =0)

lr=cls.fit(X_train, y_train)

pickle.dump(oh,open('encoder.pkl','wb'))
pickle.dump(lr,open('model.pkl','wb'))


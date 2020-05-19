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
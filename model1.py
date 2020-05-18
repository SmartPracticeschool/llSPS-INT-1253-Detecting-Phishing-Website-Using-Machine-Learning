import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
df=pd.read_csv('dataset.csv')
print(df.describe())
print(df.head())
print(df.tail())
print(df.info()) 
plt.figure(figsize =(8,8))
plt.hist(df.Result)          
  


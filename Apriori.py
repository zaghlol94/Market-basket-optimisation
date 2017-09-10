# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:24:26 2017

@author: zaghlollight
"""

#import lib and data 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
#prepare data to use it in apriori method 
transactions=[]
for i in range(dataset.shape[0]):
    li=[]
    for j in range(dataset.shape[1]):
        li.append(str(dataset.values[i,j]))
    transactions.append(li)  
#training Apriori 
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)    

res=list(rules)

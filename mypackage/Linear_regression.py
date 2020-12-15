# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:02:28 2020

@author: Patryk-PC
"""

import os
import pandas as pd
import numpy as np
from cleanedData import Cleaner
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#initialised the data cleaner class from the folder.
cleaner = Cleaner(r"D:\Uni Stuff\Modules\IN3062 Intro to AI\IN3062-Coursework")

#built the dataframe from the cleaned data.
df = cleaner.getDataFrame()

print(df[:5])

data = []
for x in df.columns:
    if x != 'price':
        data.append(x)

X = df[data].values
y = df['price'].values

print(X[:5])
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#y_train = sc.fit_transform(y_train)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print(lr_model.coef_)

y_pred = lr_model.predict(X_test)

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Amount of error: ', (np.sqrt(metrics.mean_squared_error(y_test,y_pred))/np.mean(y_test)) * 100, '%' )

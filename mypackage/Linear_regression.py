import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cleanedData import Cleaner
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics

#initialised the data cleaner class from the folder.
cleaner = Cleaner(r"D:\Uni Stuff\Modules\IN3062 Intro to AI\IN3062-Coursework")

#built the dataframe from the cleaned data.
df = cleaner.getDataFrame()

print(df[:5])

#scaler = MinMaxScaler()
#df[df.columns] = scaler.fit_transform(df[df.columns])

data = []
for x in df.columns:
    if x == 'carat': 
        data.append(x)

diamond_features = ['carat','x','y','z','color','cut','clarity']
X = df.loc[:, diamond_features].values

#X = df[data].values
#y = df['price'].values

#print(X[:5])
#print(y[:5])

df_price = np.log(df['price'])

X_train, X_test, y_train, y_test = train_test_split(X, df_price, test_size=0.20, random_state=42)

#Linear Regression Model
lr_model = LinearRegression(fit_intercept = True)
lr_model.fit(X_train, y_train)

#Polynomial Regresssion Model, Degree 2
pr_model = PolynomialFeatures(degree=2)
X_poly_train = pr_model.fit_transform(X_train)
X_poly_test = pr_model.fit_transform(X_test)
pr_model.fit(X_poly_train, y_train)
lr_model2 = LinearRegression()
lr_model2.fit(X_poly_train, y_train)

#Polynomial Regression Model, Degree 3
pr_model2 = PolynomialFeatures(degree=3)
X_poly_train2 = pr_model2.fit_transform(X_train)
X_poly_test2 = pr_model2.fit_transform(X_test)
pr_model2.fit(X_poly_train2, y_train)
lr_model3 = LinearRegression()
lr_model3.fit(X_poly_train2, y_train)

#Polynomial Regression Model, Degree 4
pr_model3 = PolynomialFeatures(degree=4)
X_poly_train3 = pr_model3.fit_transform(X_train)
X_poly_test3 = pr_model3.fit_transform(X_test)
pr_model3.fit(X_poly_train3, y_train)
lr_model4 = LinearRegression()
lr_model4.fit(X_poly_train3, y_train)

print(lr_model.coef_)

y_pred = lr_model.predict(X_test)
y_pred2 = lr_model2.predict(X_poly_test)
y_pred3 = lr_model3.predict(X_poly_test2)
y_pred4 = lr_model4.predict(X_poly_test3)

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)
print('\n')
print('Mean:', np.mean(y_test))
print('Linear Regression RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Polynomial Regression degree 2 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
print('Polynomial Regression degree 3 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))
print('Polynomial Regression degree 4 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))
#print('Amount of error: ', (np.sqrt(metrics.mean_squared_error(y_test,y_pred))/np.mean(y_test)) * 100, '%' )

#plt.scatter(X_test, y_test, c='red')
#plt.scatter(X_test,y_pred2, c='green')
#plt.show()

carat_test = X_test[:,:1]


plt.figure(1)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred, s=1, color='blue')

plt.figure(2)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred2, s=1, color='blue')

plt.figure(3)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred3, s=1, color='blue')

plt.figure(4)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred4, s=1, color='blue')
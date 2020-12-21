import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cleanedData import Cleaner
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import r2_score

#initialised the data cleaner class from the folder.
cleaner = Cleaner(r"D:\Uni Stuff\Modules\IN3062 Intro to AI\IN3062-Coursework")

#built the dataframe from the cleaned data.
df = cleaner.getDataFrame()

print(df[:5])

diamond_features = ['carat','x','y','z','color','cut','clarity','depth','table']
X = df.loc[:, diamond_features].values

#normal price value instead of log of the price.
#y = df['price'].values

#The dependent feature is the log of the price.
df_price = np.log(df['price'])

#splitting the dataset into train and test datasets.
X_train, X_test, y_train, y_test = train_test_split(X, df_price, test_size=0.25, random_state=42)

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

# =============================================================================
# #Not used because of poor results and long computational time.
# #Polynomial Regression Model, Degree 4
# pr_model3 = PolynomialFeatures(degree=4)
# X_poly_train3 = pr_model3.fit_transform(X_train)
# X_poly_test3 = pr_model3.fit_transform(X_test)
# pr_model3.fit(X_poly_train3, y_train)
# lr_model4 = LinearRegression()
# lr_model4.fit(X_poly_train3, y_train)
# =============================================================================

#The prediction values for all the models
y_pred = lr_model.predict(X_test)
y_pred2 = lr_model2.predict(X_poly_test)
y_pred3 = lr_model3.predict(X_poly_test2)

# =============================================================================
# #The predicted value for a polynomial model of degree 4
# y_pred4 = lr_model4.predict(X_poly_test3)
# =============================================================================

# =============================================================================
# #This code is to find outliers that give bad results in the polynomial regression.
# print('Error testing degree 3:')
# print(diamond_features)
# for i in range(1, len(y_pred3)):
#     if y_pred3[i] > 12:
#         print(X_test[i])
# 
# print('\n')

# print('Error testing degree 2:')
# print(diamond_features)
# for i in range(1, len(y_pred2)):
#     if y_pred2[i] > 11:
#         print(X_test[i])
# 
# print('\n')
# =============================================================================


#comparing actual to predicted values.
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Actual exp': np.exp(y_test), 'Predicted exp': np.exp(y_pred)})
df_head = df_compare.head(25)
print(df_head)
print('\n')
print('Mean:', np.mean(y_test))
print('Linear Regression RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Linear Regression R2:', r2_score(y_test,y_pred))
print('Polynomial Regression degree 2 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
print('Polynomial Regression degree 2 R2:', r2_score(y_test,y_pred2))
print('Polynomial Regression degree 3 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))
print('Polynomial Regression degree 3 R2:', r2_score(y_test,y_pred3))
# =============================================================================
# #RMSE and R2 values for polynomial regression degree 4
# print('Polynomial Regression degree 4 RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))
# print('Polynomial Regression degree 4 R2:', r2_score(y_test,y_pred4))
# =============================================================================

#Takes the carat values out of the test dataset to compare to our predicted values.
carat_test = X_test[:,:1]

#The next 4 graphs plot the predicted and actual price results against the carat value. Reason for carat in report.
#Red is actual values, blue is predicted values.
#Plots 2 scatter graphs on top of one another, results are for linear regression.
plt.figure(1)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred, s=1, color='blue')
plt.xlabel("Carat")
plt.ylabel("Log(price)")

#Plots 2 scatter graphs on top of one another, results are for polynomial regression degree 2. 
plt.figure(2)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred2, s=1, color='blue')
plt.xlabel("Carat")
plt.ylabel("Log(price)")

#Plots 2 scatter graphs on top of one another, results are for polynomial regression degree 3. 
plt.figure(3)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred3, s=1, color='blue')
plt.xlabel("Carat")
plt.ylabel("Log(price)")

# =============================================================================
# #Plots the results for polynomial regression degree 4 for the price against carat.
# plt.figure(4)
# plt.scatter(carat_test, y_test, s=1, color='red')
# plt.scatter(carat_test, y_pred4, s=1, color='blue')
# =============================================================================

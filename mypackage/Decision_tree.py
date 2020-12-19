import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cleanedData import Cleaner
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

#initialised the data cleaner class from the folder.
cleaner = Cleaner(r"D:\Uni Stuff\Modules\IN3062 Intro to AI\IN3062-Coursework")

#built the dataframe from the cleaned data.
df = cleaner.getDataFrame()

print(df[:5])

# =============================================================================
# data = []
# for x in df.columns:
#     if x == 'carat': 
#         data.append(x)
# =============================================================================

diamond_features = ['carat','x','y','z','color','cut','clarity']
X = df.loc[:, diamond_features].values
#df_price = np.log(df['price']) #This logs the price. Purpose described in the report.

#X = df[data].values
y = df['price'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#fig = plt.figure(figsize=(15,10))

#investigates the RMSE over a range of estimators plotting the result
#THIS TAKES A WHILE TO RUN!!
#This code was taken from Week 5 exercises part 2. Tweaked the code to test for RMSE and using the
#RandomForestRegressor model instead of the classifier model.
# =============================================================================
# rmse_data = []
# nums = []
# for i in range(1,128):
#     rf_model = RandomForestRegressor(n_estimators=i,criterion="mse")
#     rf_model.fit(X_train, y_train)
#     y_model = rf_model.predict(X_test)
#     rmse = np.sqrt(metrics.mean_squared_error(y_test, y_model))
#     rmse_data.append(rmse)
#     nums.append(i)
#     
# print(rmse_data)
# plt.plot(nums,rmse_data)
# plt.xlabel("Number of Trees (n_estimators)")
# plt.ylabel("RMSE")
# plt.show()
# =============================================================================

dt_model = DecisionTreeRegressor()
dt_model.fit(X_train,y_train)

rf_model = RandomForestRegressor(n_estimators=31, criterion="mse", bootstrap=True)
rf_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_dt})
df_head = df_compare.head(25)
print(df_head)
print('\n')
print('Mean:', np.mean(y_test))
print('Decision tree RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_dt)))
print('Random Forest RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))

carat_test = X_test[:,:1]


plt.figure(1)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred_dt, s=1, color='blue')

plt.figure(2)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred_rf, s=1, color='blue')
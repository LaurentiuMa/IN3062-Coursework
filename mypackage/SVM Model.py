import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mypackage.cleanedData import Cleaner
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import time

start_time = time.time()
cleaner = Cleaner(r"C:\Users\Laurentiu\OneDrive - City, University of London\IN3062 - Introduction to Artificial Intelligence\Coursework\Code\IN3062-Coursework")
df = cleaner.getDataFrame()

diamonds_features = ['carat', 'x', 'y', 'z', 'color', 'cut', 'clarity']

X = df.loc[:, diamonds_features].values
y = df.iloc[:, 6:7].values

# pca = PCA(n_components=2,whiten=True).fit(X)
# X_pca = pca.transform(X)
# print('explained variance ratio:', pca.explained_variance_ratio_)
# print('Preserved Variance:', sum(pca.explained_variance_ratio_))


X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.20)

regressor = SVR(kernel='rbf', C=50, gamma = 10)
regressor.fit(X_train, y_train)

#produce test predictions
y_pred = regressor.predict(X_test)

mean = np.mean(y_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
percent = (rmse/mean)*100

print('Mean:', mean)
print('Root Mean Squared Error:', rmse)
print('Accuracy: ' , percent)

print("--- %s seconds ---" % (time.time() - start_time))


# plt.figure(1)
# plt.scatter(X_test, y_test, c='red')
# plt.figure(1)
# plt.scatter(X_test, y_pred, c='green')


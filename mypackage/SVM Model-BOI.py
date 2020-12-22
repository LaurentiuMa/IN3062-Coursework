import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mypackage.cleanedData import Cleaner
from mypackage.Hyperparam import Hyperparam_collection
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR
import time
from sklearn.metrics import r2_score

# Use the Cleaner module to import the dataset fom the specified raw string
cleaner = Cleaner(r"C:\Users\Laurentiu\OneDrive - City, University of London\IN3062 - Introduction to Artificial Intelligence\Coursework\Code\IN3062-Coursework")
df = cleaner.getDataFrame()

# Here we have the features we would like to include in the model, we set the X and the logY of the price
diamonds_features = ['carat', 'x', 'y', 'z', 'color', 'cut', 'clarity', 'depth', 'table']
X = df.loc[:, diamonds_features].values
y = np.log(df.iloc[:, 6:7].values)

# Set the parameters you would like to run through. If you go ahead with only 1 set, specify them
# and the program will still run as intended
kernels = ["rbf"]
pca_components = [-1]
c_value = [20]
gamma_value = [0.1]
epsilon_value = [0.1]

# Set the path where you want the results to be collected (does not work well with cloud drives)
results_file = r"C:\Users\Laurentiu\SVM_Results(non-PCA).txt"
# calculate the total number of hyperparameter combinations.
total_epochs = len(pca_components) * len(c_value) * len(gamma_value) * len(epsilon_value) * len(kernels)
epoch_number = 0
hyperparam_list = []

# Adds 5 empty collections so that you can output the top 5 results without any unnecessary checks
for j in range(5):
    hyperparam_list.append(Hyperparam_collection(0,0,0,0,0,0,0,0,0))

print(total_epochs)
print('\n')

# Run through the model with every single combination expressed above
for k in kernels:
    for component in pca_components:
        for c in c_value:
            for g in gamma_value:
                for e in epsilon_value:
                    start_time = time.time()
                    
                    # The file is opened and closed every epoch so that the state is saved every time a new epoc is run.
                    # This acts as a safety measure in the event that the program crashes unexpectedly.
                    file = open(results_file, "a")
                    file.write("\n")
                    
                    # pca = PCA(n_components=component,whiten=True).fit(X)
                    # X_pca = pca.transform(X)
    
                    # Training/testing split
                    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.20)
                    
                    
                    # Define the regressor and fit the data
                    regressor = SVR(kernel=k, C=c, gamma=g, epsilon=e)
                    regressor.fit(X_train, y_train)
    
                    #produce test predictions
                    y_pred = regressor.predict(X_test)
                    
                    end_time = time.time() - start_time
                    epoch_number += 1
                    
                    # Accuracy metrics
                    mean = np.mean(y_test)
                    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                    percent_rmse = (rmse/mean)*100
                    r_squared = r2_score(y_test, y_pred)
                    
                    # Write the hyperparameters and metrics to the text file and close it to save.
                    file.write("\n")
                    file.write("Hyperparameters: PCA-components = {0},  C = {1},  Gamma = {2},  Epsilon = {3}, Kernel = {4}".format(component,c,g,e,k))
                    file.write("\n")
                    file.write("Mean: {0}".format(mean))
                    file.write("\n")
                    file.write("RMSE: {0}".format(rmse))
                    file.write("\n")
                    file.write("% RMSE: {0}".format(percent_rmse))
                    file.write("\n")
                    file.write("R^2 score: {0}".format(r_squared))
                    file.write("\n")
                    file.write("time taken: {0}".format(end_time))
                    file.write("\n")
                    file.write("================================================================")
                    file.close()
                    
                    hyperparam_list.append(Hyperparam_collection(c, component, e, g, rmse, percent_rmse, r_squared, end_time, k))
                    
                    # Print to console to ensure that the program is working
                    percentage_completed = (epoch_number/total_epochs)*100
                    print("epoch {0}/{1}".format(epoch_number, total_epochs))
                    
                    # Sort the list by R^2 and print the 5 best collections of hyperparameters
                    hyperparam_list.sort(key=lambda ac: ac.rsquared, reverse=True)
                    for i in range(5):
                        h_item = hyperparam_list[i]
                        print("R^2 = {0}, PCA-components = {1},  C = {2},  Gamma = {3},  Epsilon = {4}, Exec-duration = {5}s, kernel = {6}".format(h_item.rsquared ,h_item.components,h_item.c,h_item.gamma,h_item.epsilon, h_item.time_executing, h_item.kernel_used))
                        
                    print("{0}% completed".format(percentage_completed))
                    print('\n')

                    plt.figure(0)
                    plt.ylabel('log(Price)')
                    plt.xlabel('carat')
                    plt.scatter(X_test[:,:1], y_test, c='blue')
                    plt.scatter(X_test[:,:1], y_pred, c='red')
                    
                    plt.figure(1)
                    plt.ylabel('Price')
                    plt.xlabel('carat')
                    plt.scatter(X_test[:,:1], np.exp(y_test), c='blue')
                    plt.scatter(X_test[:,:1], np.exp(y_pred), c='red')
                
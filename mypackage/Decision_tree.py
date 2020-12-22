import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cleanedData import Cleaner
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
from sklearn.metrics import r2_score

def HyperParameterTraining(path):
    # Change this path to a path relative to where u want to store the results.
    results_file = path

    max_depth_vals = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    min_sample_split_vals = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    best_r2_score = 0
    best_max_depth = 0
    best_min_split_sample = 0

    for d in max_depth_vals:
        for sample in min_sample_split_vals:
            dt_model = DecisionTreeRegressor(criterion= "mse", random_state= 42, 
                                             max_depth= d, min_samples_split= sample)
            dt_model.fit(X_train,y_train)
            
            y_pred_dt = dt_model.predict(X_test)
            print('\n')
            print("max_depth = {0}, min_sample_split = {1}".format(d, sample))
            print('Mean:', np.mean(y_test))
            print('Decision tree RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_dt)))
            r2 = r2_score(y_test,y_pred_dt)
            print('Decision tree R2:', r2)
            
            if r2 > best_r2_score:
                best_r2_score = r2
                best_max_depth = d
                best_min_split_sample = sample
                
            file = open(results_file, "a")
            file.write("\n")
            file.write("max_depth = {0}, min_sample_split = {1}".format(d, sample))
            file.write("\n")
            file.write("Mean: {0}".format(np.mean(y_test)))
            file.write("\n")
            file.write("Decision tree RMSE: {0}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_dt))) )
            file.write("\n")
            file.write("Decision tree R2: {0}".format(r2_score(y_test,y_pred_dt)))
            file.write("\n")
            file.write("================================================================")
            file.close()
                
    print('\n')
    print("==========================================================================")
    print("The best results are:")
    print("r2_score = {0}, max_depth = {1}, min_split_sample = {2}".format(best_r2_score, best_max_depth, best_min_split_sample))
    print("==========================================================================")
                
    return


def RandomForestHyperParameterTraining(max_n):
    # investigates the RMSE over a range of estimators plotting the result
    # This code might take a while to run depending on the value of max_n
    # This code was taken from Week 5 exercises part 2. Tweaked the code to test for RMSE and using the
    # RandomForestRegressor model instead of the classifier model.
    
    fig = plt.figure(figsize=(15,10))
    
    rmse_data = []
    nums = []

    for i in range(1,max_n + 1):
        rf_model = RandomForestRegressor(n_estimators=i, max_depth=13, min_samples_split=13,
                                         criterion="mse", random_state=42)
        rf_model.fit(X_train, y_train)
        y_model = rf_model.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_model))
        rmse_data.append(rmse)
        nums.append(i)

    print(rmse_data)
    plt.plot(nums,rmse_data)
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("RMSE")
    plt.show()

    return

def RandomForestHyperparameterTrainingFull():
    # This code would find the optimal values for max_depth, min_sample_split and the n_estimators
    # Takes an extremely long time to compute, due to time constraints I had to test it with the values shown.
    # Changine the values of the first 3 variables in this function would control for how many iterations the model is trained
    
    max_depth_vals = [10,11,12,13,14,15]
    min_sample_split_vals = [10,11,12,13,14,15]
    estimators = 50
    
    best_r2_score = 0
    best_max_depth = 0
    best_min_split_sample = 0
    best_n_estimators = 0;
    
    total_epochs = len(max_depth_vals) * len(min_sample_split_vals) * estimators
    current_epoch = 0
    
    for i in range(1, estimators + 1):
        for d in max_depth_vals:
            for sample in min_sample_split_vals:
                rf_model = RandomForestRegressor(n_estimators=i, max_depth=d, min_samples_split=sample,
                                                 criterion="mse", random_state=42)
                rf_model.fit(X_train, y_train)
                y_model = rf_model.predict(X_test)
                r2 = r2_score(y_test, y_model)
                if r2 > best_r2_score:
                    best_r2_score = r2
                    best_max_depth = d
                    best_min_split_sample = sample
                    best_n_estimators = i
                    
                current_epoch += 1
                completed = (current_epoch/total_epochs) * 100
                print ("{0}% completed".format(completed))
    
    print('\n')
    print("==========================================================================")
    print("The best results are:")
    print("r2_score = {0}, max_depth = {1}, min_split_sample = {2}, n_estimators = {3}".format(best_r2_score, best_max_depth, best_min_split_sample,best_n_estimators))
    print("==========================================================================")
    
    return



# Use the Cleaner module to import the dataset fom the specified raw string, place in dataframe.
cleaner = Cleaner(r"D:\Uni Stuff\Modules\IN3062 Intro to AI\IN3062-Coursework")
df = cleaner.getDataFrame()

# List of features that we are analysing
diamond_features = ['carat','x','y','z','color','cut','clarity','depth','table']
X = df.loc[:, diamond_features].values

# This logs the price. Purpose described in the report.
df_price = np.log(df['price'])

# Splitting the dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, df_price, test_size=0.20, random_state=42)

# Uncomment this function call for the Decision Tree Hyperparameter testing, change the path for where you want to store the .txt file.
#HyperParameterTraining(r"D:\Hyperparam Training\Decision_Tree_Results.txt")

# Initialising the model based on the hyperparameter training and fitting the model to the dataset
dt_model = DecisionTreeRegressor(criterion= "mse", random_state= 42, 
                                 max_depth= 13, min_samples_split= 13)
dt_model.fit(X_train,y_train)

# Uncomment this function call for the Random Forest n_estimator testing, change the parameter to set the max n_estimator value for training.
#RandomForestHyperParameterTraining(100)

# Uncomment this function call for the full Random Forest hyperparameter testing, This test takes a really long time.
# Works out the max_depth, min_sample_split and n_estimators parameters for the Random Forest model.
#RandomForestHyperparameterTrainingFull()

# Initialising the model based on the hyperparameter training and fitting the model to the dataset
rf_model = RandomForestRegressor(n_estimators=47, max_depth=15, min_samples_split=11,
                                 criterion="mse", bootstrap=True, random_state=42)
rf_model.fit(X_train, y_train)

# Using the testing dataset to predict the price of the diamonds.
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Visual comparison of the actual and predicted values.
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_dt, 'Actual exp': np.exp(y_test), 'Predicted exp': np.exp(y_pred_dt)})
df_head = df_compare.head(25)
print(df_head)

# Printing the mean, RMSE for both the Decision Tree and Random Forest model and the R2 values for both models.
print('\n')
print('Mean:', np.mean(y_test))
print('Decision tree RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_dt)))
print('Decision tree R2:', r2_score(y_test,y_pred_dt))
print('Random Forest RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
print('Random Forest R2:', r2_score(y_test,y_pred_rf))

carat_test = X_test[:,:1]

# Plotting a scatter diagram of actual test data and the predicted values from the Decision Tree model based on the carat feature.
plt.figure(1)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred_dt, s=1, color='blue')
plt.xlabel("Carat")
plt.ylabel("Log(price)")

# Plotting a scatter diagram of actual test data and the predicted values from the Random Forest model based on the carat feature.
plt.figure(2)
plt.scatter(carat_test, y_test, s=1, color='red')
plt.scatter(carat_test, y_pred_rf, s=1, color='blue')
plt.xlabel("Carat")
plt.ylabel("Log(price)")



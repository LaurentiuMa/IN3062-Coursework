# insert imports here
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from mypackage.cleanedData import Cleaner
import matplotlib.pyplot as plt

# Retrieve cleaned data from lauri's module and create dataframe
# Make sure to change the working directory if running on another machine
cleaner = Cleaner(r"/home/ismael/Documents/Uni/Intro to AI/Coursework/IN3062-Coursework/mypackage")
df = df = cleaner.getDataFrame()

# Sort data
diamonds_features = ['carat', 'x', 'y', 'z', 'color', 'cut', 'clarity']
X = df.loc[:, diamonds_features].values
y = np.log(df.iloc[:, 6:7].values)

# Method that runs the training
def knn_hyper_param_train():
    n_choices = [2, 4, 8, 16, 32, 64]
    
    file = open("KNN_results.txt", "a")
    file.write("\n")
    
    for nc in n_choices:
        
        loop_RMSE = np.array([])
        loop_mean = np.array([])
        loop_r2_score = np.array([])
        
        for l in range(3):       
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            knn_model = KNeighborsRegressor(n_neighbors=nc)
            knn_model.fit(X_train, y_train)
            pred = knn_model.predict(X_test)
            score = np.sqrt(metrics.mean_squared_error(y_test, pred))
            r2_score = metrics.r2_score(y_test, pred)
            print("------------loop {0}----------------\n".format(l))
            print("RMSE: {0}\n".format(score))
            print("Final score (RMSE): {0}\n".format(score))
            print("Mean: {}\n".format(np.mean(y_test)))
            print("Accuracy: {}\n".format((score/np.mean(y_test)) * 100))
            print("------------------------------------")
            loop_r2_score = np.append(loop_r2_score, r2_score)
            loop_RMSE = np.append(loop_RMSE, score)
            loop_mean = np.append(loop_mean, np.mean(y_test))
        
        file.write("----------- Loop averages ------------\n")
        file.write("Neighbors checked: {0}\n".format(nc))
        file.write("R2 score: {0}\n".format(loop_r2_score.mean()))
        file.write("Average RMSE: {0}\n".format(loop_RMSE.mean()))
        file.write("Average mean: {0}\n".format(loop_mean.mean()))
        file.write("Average prediction accuracy: {0}\n".format((loop_RMSE.mean()/loop_mean.mean()) * 100))
        file.write("--------------------------------------\n")
        print("----------- Loop averages ------------\n")
        print("Neighbors checked: {0}\n".format(nc))
        print("R2 score: {0}".format(loop_r2_score.mean()))
        print("Average RMSE: {0}\n".format(loop_RMSE.mean()))
        print("Average mean: {0}\n".format(loop_mean.mean()))
        print("Average prediction accuracy: {0}\n".format((loop_RMSE.mean()/loop_mean.mean()) * 100))
        print("--------------------------------------\n")
    
    file.close()

#knn_hyper_param_train()

def test_knn(nc):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    knn_model = KNeighborsRegressor(n_neighbors=nc)
    knn_model.fit(X_train, y_train)
    pred = knn_model.predict(X_test)
    score = np.sqrt(metrics.mean_squared_error(y_test, pred))
    r2_score = metrics.r2_score(y_test, pred)
    
    print("R2 score: {0}".format(r2_score))
    print("RMSE: {0}".format(score))
    print("Mean: {0}".format(np.mean(y_test)))
    
    plt.scatter(X_test[:, 0], pred, c='#FF0000')
    plt.scatter(X_test[:, 0], y_test, c='#0000FF')
    plt.xlabel("carat")
    plt.ylabel("log(price)")
    
test_knn(4)
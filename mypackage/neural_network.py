# insert imports here
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.utils
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.models import load_model
import os
from mypackage.cleanedData import Cleaner
import matplotlib.pyplot as plt

# Retrieve cleaned data from cleaner module and create dataframe
# Make sure to change the working directory if running on another machine
cleaner = Cleaner(r"/home/ismael/Documents/Uni/Intro to AI/Coursework/IN3062-Coursework/mypackage")
df = df = cleaner.getDataFrame()

# Sort data
diamonds_features = ['carat', 'x', 'y', 'z', 'color', 'cut', 'clarity']
X = df.loc[:, diamonds_features].values
y = np.log(df.iloc[:, 6:7].values)

# Method that runs the hyper-parameter training
def nn_hyper_param_training():
    
    # Dictionary used to store all other values as a tuple key with the R2 score as the value, used to order from highest R2 score in txt file later
    values_to_r2 = {}
    # Store final results
    r_squared = []
    final_RMSE = []
    final_mean = []
    final_acc = []
    
    # All possible hyper-parameter values
    epoch_choices = [16, 32, 64]
    node_choices = [32, 64, 128]
    lr_choices = [0.001, 0.002, 0.004]
    
    # UNCOMMENT TO LOAD A PREVIOUSLY SAVED MODEL, FOR TESTING ONLY
    #model2 = load_model(os.path.join(save_path,"network.h5")) #===If USING SAVED MODEL UN-COMMENT AND PUT IN LOOP===
    
    # Used to keep track of how long the training has run for
    start_total_time = time.time()
    print("--- Total timer starts now... ---")
    
    # Nested for loops to ensure every combination of hyper-parameter values
    for EPOCHS in epoch_choices:
        for nodes in node_choices:
            for lr in lr_choices:
                # Open txt file to store all the values to
                file = open("3_relu_results.txt", "a")
                # Used to format file and console
                file.write("==================================================================================\n")
                print("==================================================================================\n")
                file.write("Starting loop for with:\nEpochs: {0}\nNodes: {1}\nLearning Rate: {2}\n".format(EPOCHS, nodes, lr))
                print("Starting loop for with:\nEpochs: {0}\nNodes: {1}\nLearning Rate: {2}\n".format(EPOCHS, nodes, lr))
                # Store results of current loop to work out average after loop
                loop_RMSE = np.array([])
                loop_mean = np.array([])
                loop_r2_score = np.array([])
                # Keep track of how long each loop took
                start_loop_time = time.time()
                # We run three loops to ensure fair results using the average
                for l in range(3):
                    # Split the dataframe
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
                    
                    # Creating the neural netwrok model
                    nn_model = Sequential()
                    # Adding the hidden layers to the model
                    nn_model.add(Dense(nodes, input_shape=X[1].shape, activation='relu'))
                    nn_model.add(Dense(nodes, activation='relu')) 
                    nn_model.add(Dense(nodes, activation='relu'))
                    # Adding the output layer to the model
                    nn_model.add(Dense(1))
                    nn_model.summary()
                    
                    # Adding he optimiser to the model
                    custom_adam = tensorflow.keras.optimizers.Adam(learning_rate=lr)
                    nn_model.compile(loss='mean_squared_error', optimizer=custom_adam)
                    # Training the model using the training data
                    nn_model.fit(X_train,y_train,verbose=2, epochs=EPOCHS)
                    nn_model.summary()
                    
                    # Testing the model
                    pred = nn_model.predict(X_test)
                    score = np.sqrt(metrics.mean_squared_error(y_test, pred))
                    r2_score = metrics.r2_score(y_test, pred)
                    
                    # Printing results to console and writing results to txt file
                    file.write("------ Loop {0} ---------------\n".format(l+1))
                    print("RMSE: {0}".format(score))
                    file.write("RMSE: {0}\n".format(score))
                    loop_RMSE = np.append(loop_RMSE, score)
                    print("Mean: {}".format(np.mean(y_test)))
                    file.write("Mean: {0}\n".format(np.mean(y_test)))
                    loop_mean = np.append(loop_mean, np.mean(y_test))
                    print("Prediction accuracy: {}%".format((score/np.mean(y_test)) * 100))
                    file.write("Prediction accuracy: {0}\n".format((score/np.mean(y_test)) * 100))
                    loop_r2_score = np.append(loop_r2_score, r2_score)
                    print("R2 score: {0}\n".format(r2_score))
                    file.write("R2 score: {0}\n".format(r2_score))
                
                # Printing writing average ofresults to txt file
                file.write("----------- Loop averages ------------\n")
                file.write("R2 score: {0}".format(loop_r2_score.mean()))
                file.write("Average RMSE: {0}\n".format(loop_RMSE.mean()))
                file.write("Average mean: {0}\n".format(loop_mean.mean()))
                file.write("Average prediction accuracy: {0}\n".format((loop_RMSE.mean()/loop_mean.mean()) * 100))
                # Printing average of results to console
                file.write("--------------------------------------\n")
                print("----------- Loop averages ------------\n")
                print("R2 score: {0}".format(loop_r2_score.mean()))
                print("Average RMSE: {0}\n".format(loop_RMSE.mean()))
                print("Average mean: {0}\n".format(loop_mean.mean()))
                print("Average prediction accuracy: {0}\n".format((loop_RMSE.mean()/loop_mean.mean()) * 100))
                print("--------------------------------------\n")
                print("=====================Time taken: {0} Mins======================\n".format(round((time.time() - start_loop_time)/60, 5)))
                file.write("==================================================================================\n")
                # Closing txt file in order for the file to change
                file.close()
                # Adding all the values to the final results arrays
                r_squared.append(loop_r2_score.mean())
                final_RMSE.append(loop_RMSE.mean())
                final_mean.append(loop_mean.mean())
                final_acc.append((loop_RMSE.mean()/loop_mean.mean()) * 100)
                # Adding new features and results to dictionary to sort later
                values_to_r2[(EPOCHS, nodes, lr, loop_RMSE.mean(), loop_mean.mean(), (loop_RMSE.mean()/loop_mean.mean()) * 100)] = loop_r2_score.mean()
    
    # Print total time taken
    print("--- Total time: {} mins ---".format(round((time.time() - start_total_time)/60, 5)))
    
    # Sort R2 array from largest to smallest
    r_squared.sort(reverse=True)
    # Open txt file which will store all results based on the order of the R2 score array (largest to smallest)
    file = open("3_relu_top_results.txt", "a")
    file.write("\n")
    # Loop through values in the R2 score array and find the the corresponding key in the dictionary to write to the text file in the correct order
    for r in r_squared:
        for k, v in values_to_r2.items():
            # Write the hyper-parameter value and results to the txt file if the corresponding key was found
            if v == r:
                file.write("==================================================================================\n")
                file.write("Epochs: {0}\nNodes per layer: {1}\nLearning rate: {2}\n".format(k[0], k[1], k[2]))
                file.write("-------------Stats---------------\n")
                file.write("R squared score: {0}\n".format(r))
                file.write("RMSE: {0}\nMean: {1}\nPrediction Accuracy: {2}\n".format(k[3], k[4], 100 - k[5]))
                file.write("==================================================================================\n")
                # Break to save computation resources in case the value was found early in the dictionary,
                # the chances of two R2 scores being the same is very low and would be noticed in the xt file if occured
                break
    # Close the file in order for the txt file to update
    file.close()
    
    # UNCOMMENT TO SAVE THE MODEL BEING TRAINED
    #save_path = "."
    #nn_model.save(os.path.join(save_path,"3_relu_model.h5"))

# Method to test the NN model with set hyper-parameter, also displays scatter graph of results
def test_nn(EPOCHS, nodes, lr):
    # Split dataframe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
                    
    # Creating the neural netwrok model
    nn_model = Sequential()
    # Adding the hidden layers to the model
    nn_model.add(Dense(nodes, input_shape=X[1].shape, activation='relu'))
    nn_model.add(Dense(nodes, activation='relu')) 
    nn_model.add(Dense(nodes, activation='relu'))   
                     # Adding the output layer tot he model 
    nn_model.add(Dense(1))
    nn_model.summary()
    
    # Adding he optimiser to the model
    custom_adam = tensorflow.keras.optimizers.Adam(learning_rate=lr)    
    nn_model.compile(loss='mean_squared_error', optimizer=custom_adam)
    # Training the model using the training data
    nn_model.fit(X_train,y_train,verbose=2, epochs=EPOCHS)
    nn_model.summary()
    
    # Testing the model
    pred = nn_model.predict(X_test)
    score = np.sqrt(metrics.mean_squared_error(y_test, pred))
    r2_score = metrics.r2_score(y_test, pred)
    
    # Print the results to the console
    print("R2 score: {0}".format(r2_score))
    print("RMSE: {0}".format(score))
    print("Mean: {0}".format(np.mean(y_test)))

    # Display the scatter graphs comparing the true and predicted lop(price) against "carat"
    plt.scatter(X_test[:, 0], pred, c='#FF0000')
    plt.scatter(X_test[:, 0], y_test, c='#0000FF')
    plt.xlabel("carat")
    plt.ylabel("log(price)")

#nn_hyper_param_training()
#test_nn(64, 64, 0.001)


class Hyperparam_collection:
    c = 0
    components = 0
    epsilon = 0
    gamma = 0
    rmse = 0
    accuracy = 0
    rsquared = 0
    time_executing = 0
    kernel_used = ""
    
    def __init__(self,c,components,epsilon,gamma,rmse,accuracy,rsquared,time_executing, kernel_used):
        self.c = c
        self.components = components
        self.epsilon = epsilon
        self.gamma = gamma
        self.rmse = rmse
        self.accuracy = accuracy
        self.rsquared = rsquared 
        self.time_executing = time_executing
        self.kernel_used = kernel_used
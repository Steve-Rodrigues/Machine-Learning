import numpy as np
import pandas as pd

ice_cream_data = pd.read_csv('./ice_cream_sales.csv')
print(ice_cream_data.head())
features = ice_cream_data[['temperature_f','foot_traffic','is_weekend']]
X = features.to_numpy() #creates the matrix of features (coefficient matrix)
X = (X-np.mean(X,axis=0))/np.std(X,axis=0) #scaling the features so they are the same scale when we do the learning step
Y = np.ravel(ice_cream_data['revenue']) #the label vector (solution col-vector)
weights = np.zeros(3) #these are the weights which are going to be changed as the model adjusts to the data set while we minimize the cost 
b = 0 #this is the bias (the y-intercept to adjust the hypothesis independent of the other weights)
learning_step = 0.1 #how big of a step we take at each gradient descent
tolerance = 1e-6 #checking that the cost went down by at least this much each time else we exit because it converged

#now lets minimize the cost funciton (avg error) of the hypothesis by looping through it until we reach max iterations or the cost converges(levels out)
max_iterations = 1000 #1000 cost minimizes to start
#here are the initial values for the hypothesis and cost
hypothesis = X@weights + b #matrix mult for the dot product and add b(slope to each). Same as looping through and doing each instance
error = hypothesis - Y #vector of how off each predicition was in the hypothesis (200x1) matrix
prevCost = np.mean(error**2) #gives the first cost (very high to start because weights were 0 so the mapping was bad)
print(f'*********Initial RMSE: {np.sqrt(prevCost)}')
#minimize the cost by looping max-iterations times or till the cost converges(levels out ) to find the optimal weights for the hypothesis
for i in range(max_iterations):
    rows, cols = np.shape(X)
    partialsW = np.zeros(cols)
    partialB = np.mean(error) #partial derivative of the bias 
    for i in range(cols):
        partialsW[i] = np.dot(error, X[:,i]) / rows #stores the avg of the error scaled by one of the features 
    #now perform gradient descent to update the weights
    for i in range(cols):
        weights[i] = weights[i] - (learning_step*partialsW[i])
    b = b - (learning_step*partialB) #update the bias with gradient descent
    hypothesis = X@weights + b
    error = hypothesis - Y
    cost = np.mean(error**2)

    if abs(prevCost - cost) < tolerance:
        break
    prevCost = cost
print(f"Final result is that the model is now off by {np.sqrt(cost)}")
print(f"The best fit equation found for the hypothesis function: {weights[0]}x1 + {weights[1]}x2 + {weights[2]}x3 + {b}")
testing = pd.read_csv('./ice_cream_sales_test.csv')
X_test_features = testing[['temperature_f','foot_traffic', 'is_weekend']]
X_test_labels = np.ravel(testing['revenue'])
X_test_scaled = (X_test_features - np.mean(X_test_features, axis=0)) / np.std(X_test_features, axis=0)
def test(X,Y):
    predict = X@weights + b
    err = predict - Y
    cost = np.mean(err**2)
    rmse  = np.sqrt(cost)
    return f"The testing data predictions were off by {rmse} on average for all of the instances and their labels"
print(test(X_test_scaled, X_test_labels))
    




    

    



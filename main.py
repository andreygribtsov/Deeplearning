#%%
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()
n_hidden_neurons_qty = 5
n_output_neurons_qty = 1

print("X.shape = {}".format(X.shape))
print("Y.shape = {}".format(Y.shape))

# visualize the data
# plt.scatter(X[0, :], X[1, :], c= Y, s=40, cmap=plt.cm.Spectral)

# # train the logistic regression classifier
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)

# # plot the decision boundary for logistic regression
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)

# # print accuracy
# LR_predictions = clf.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#        '% ' + "(percentage of correctly labelled datapoints)")

# MODEL WITH NEURAL NETWORK WITH ONE HIDDEN LAYER

# initialize weigth and biases for NN model 1

W1 = np.random.randn(X.shape[0], n_hidden_neurons_qty)
b1 = np.random.randn(1, n_hidden_neurons_qty)
W2 = np.random.randn(n_hidden_neurons_qty, n_output_neurons_qty)
b2 = np.random.randn(1, n_output_neurons_qty)

print("W1 = \n {}".format(W1))
print("b1 = \n {}".format(b1))
print("W2 = \n {}".format(W2))
print("b2 = \n {}".format(b2))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def hyper_tan(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

# forward propogation

Z1 = W1.T @ X + b1.T
A1 = hyper_tan(Z1)
Z2 = W2.T @ A1 + b2.T
Y2 = sigmoid(Z2)

cost = (-1) * np.sum((Y * np.log(Y2) + (1 - Y) * np.log(1 - Y2)))

# back propagation

learning_rate = 0.005
iteration_qty = 10000

# print("W2 = {}\n".format(W2.shape))
# print("Y2 - Y = {}\n".format((Y2 - Y).shape))
costs = []

for i in range(0, iteration_qty):

    dZ1 = (Y2 - Y)
    dcost_dW2 = (dZ1 @ A1.T)
    W2 = W2 - learning_rate * dcost_dW2.T
    dcost_b2 = np.sum(dZ1)
    b2 = b2 - learning_rate * dcost_b2

    dcost_dW1 = W2 @ dZ1 * (1 - np.power(A1, 2)) @ X.T
    W1 = W1 - learning_rate * dcost_dW1.T
    dcost_b1 = np.sum(W2 @ dZ1 * (1 - np.power(A1, 2)))
    b1 = b1 - learning_rate * dcost_b1 

    Z1 = W1.T @ X + b1.T
    A1 = hyper_tan(Z1)
    Z2 = W2.T @ A1 + b2.T
    Y2 = sigmoid(Z2)
    cost = (-1) * np.sum((Y * np.log(Y2) + (1 - Y) * np.log(1 - Y2)))
    if i % 100 == 0:
        costs.append(i, cost)

# plot the cost
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (x1,000)')
plt.show()

print("learning results \n")
print("W1 = {} \n".format(W1))
print("b1 = {} \n".format(b1))
print("W2 = {} \n".format(W2))
print("b2 = {} \n".format(b2))

# normalize what is predicted

results = np.zeros([1, X.shape[1]])

for i in range(0, X.shape[1]):
    results[0, i] = 1 if Y2[0, i] > 0.5 else 0

print(
    "train accuracy {} %".format(
    100 - np.mean(np.abs(results - Y)) * 100
    )
)





# %%

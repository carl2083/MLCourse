import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def computeCostMulti(X,y,theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp,2)) /(2*m)
def gradientDescentMulti (X,y,theta,alpha,iterations):
    #J_history = []
    m = X.shape[0]
    for i in range(iterations):
        ## code
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T,temp)
        theta = theta - (alpha / m) * temp
        #J_history.append(computeCostMulti(X, y, theta))
    return theta


ex1data2_path = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex1\ex1\ex1data2.txt"
data = np.loadtxt(ex1data2_path,delimiter = ',')
X = data[:,0:2]
X = (X - np.mean(X))/np.std(X)
#X[:,0] = X[:,0] / 10000
y = data[:,2]
m = data.shape[0]
ones = np.ones((m,1))
X = np.hstack((ones,X))
alpha = 0.01
num_iters = 400
theta = np.zeros((3,1))
y = y[:,np.newaxis]

J = computeCostMulti(X,y,theta)
print(J)

theta = gradientDescentMulti(X,y,theta,alpha,num_iters)
print(theta)
J = computeCostMulti(X,y,theta)
print(J)


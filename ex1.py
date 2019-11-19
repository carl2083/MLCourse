import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost ( X, y, theta):
    m = y.size
    J = 0
    J = 0.5 / m * (X.dot(theta[np.newaxis].T ) - y) ** 2
    return J
def gradientDescent (X, y , theta, alpha, num_iters):
    J_history = []
    m = y.size
    for i in range (num_iters):
        ## code
        theta = theta - alpha / m * np.sum((X.dot(theta[np.newaxis].T)-y).dot(X))

        J_history.append(computeCost(X,y,theta))
    return theta, J_history
ex1data1_path = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex1\ex1\ex1data1.txt"
ex1data2_path = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex1\ex1\ex1data2.txt"
data = np.loadtxt(ex1data1_path,delimiter = ',')
m = data.shape[0]
X = np.vstack(zip(np.ones(m),data[:,0]))
y = data[:,1]
m = len(y)
theta = np.zeros(2)
#data.head()

J = computeCost(X,y,theta)
print(J)

iterations = 1600
alpha = 0.01

theta, J_history = gradientDescent(X, y , theta, alpha, iterations)
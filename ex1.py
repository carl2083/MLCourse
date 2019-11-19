import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost ( X, y, theta):
    m = y.size
    J = 0
    J =  np.sum(np.power((X.dot(theta) - y),2)) / (2*m)
    return J
def gradientDescent (X, y , theta, alpha, num_iters):
    J_history = []
    m = y.size
    for i in range (num_iters):
        ## code
        theta = theta - alpha / m * np.sum(np.multiply((X.dot(theta)-y) , X[:,1][:,np.newaxis]))

        J_history.append(computeCost(X,y,theta))
    return theta, J_history

ex1data1_path = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex1\ex1\ex1data1.txt"
ex1data2_path = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex1\ex1\ex1data2.txt"
data = np.loadtxt(ex1data1_path,delimiter = ',')
m = data.shape[0]
X = np.vstack(list(zip(np.ones(m),data[:,0])))
y = data[:,1]
y = y[:,np.newaxis]
m = len(y)
theta = np.zeros(2)
theta = theta[:,np.newaxis]
#data.head()

J = computeCost(X,y,theta)
print(J)

iterations = 1600
alpha = 0.01

theta, J_history = gradientDescent(X, y , theta, alpha, iterations)

print(theta[0], theta[1])

plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()
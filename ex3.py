import scipy
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costFunction(X, y, theta, lmbda):
    m = len(y)
    hx = sigmoid(X @ theta)
    reg = lmbda/(2*m) * (theta[1:] @ theta[1:].T)
    J = 1 / m * np.sum(-y * np.log(hx) - (1 - y) * np.log(1 - hx)) + reg
    return J

def gradient(X, y, theta, lmbda):
    m = len(y)
    hx = sigmoid(X @ theta)
    reg = lmbda/m * theta
    grad = 1 / m * (X.T @ (hx - y))
    grad[0] = grad[0] - theta[0] * lmbda/m
    return grad

ex3data1_path = r"C:\Users\carlc\Desktop\MLcourse\machine-learning-ex3\ex3\ex3data1.mat"

data = loadmat(ex3data1_path)
X = data['X']
y = data['y']

# _, axarr = plt.subplots(10,10,figsize=(10,10))
# for i in range (10):
#     for j in range (10):
#         axarr[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((20,20),order = 'F'))
#         axarr[i,j].axis('off')

m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones,X))
(m,n) = X.shape
lmbda  = 0.1
k = 10

theta = np.zeros((k,n))
for i in range (k):
    digit_class = i if i else 10
    theta[i] = opt.fmin_cg(f= costFunction,x0=theta[i],fprime=gradient,args=(X,(y == digit_class).flatten(),lmbda),maxiter = 50)

pred = np.argmax(X @ theta.T, axis = 1)
pred = [e if e else 10 for e in pred]
np.mean(pred == y.flatten()) * 100

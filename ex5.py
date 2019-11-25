import numpy as np
import scipy.io

datafile = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex5\ex5\ex5data1.mat"

mat = scipy.io.loadmat(datafile)
X = mat['X']
y = mat['y']
Xtest = mat['Xtest']
ytest= mat['ytest']
Xval= mat['Xval']
yval= mat['yval']
X = np.insert(X,0,1,axis=1)
def computeCost(theta, X, y, mylambda=0.):
    m = X.shape[0]
    return 1 / (2*m) * np.sum(np.square(X @ theta - y)) + mylambda / (2 * m) * theta[1:-1,:].T @ theta[1:-1,:]
theta = np.ones(2)
theta = theta[:,np.newaxis]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def gradient(theta, X,y, mylambda=0.):
    m = X.shape[0]
    grad = 1/m * (X.T @ (X @ theta - y)) + mylambda / m * theta
    grad[0] = grad[0] - mylambda / m * theta[0]
    return grad
def trainingError(theta, X, y):
    return 1 / (2*X.shape[0]) * np.sum(np.square(X @ theta - y))

def degreeOfX(X,degree):
    if degree == 1:
        return X
    stack = X
    for i in range (2, degree + 1):
        stack = np.hstack((stack, X ** i))
    return stack




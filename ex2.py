import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
ex2data1_path = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex2\ex2\ex2data1.txt"
data = np.loadtxt(ex2data1_path, delimiter=',')


def sigmoid(a):
    return 1/(1 + pow(np.e, -a))


def computeCost (theta,X,y,lbda):
    print(theta.shape)
    m = X.shape[0]
    hx = sigmoid(X @ theta)
    J = (1/m) * np.sum(np.multiply(-y, np.log(hx)) - np.multiply ((1-y),np.log(1-hx)))
    reg = (lbda/(2*m)) * theta[1:].T @ theta[1:]
    J = J + reg
    return J
def gradient(theta,X,y, lbda):
    m = X.shape[0]
    grad = ((1/m) * (X.T @ (sigmoid(X @ theta) - y))) + lbda/m * theta
    theta[0] = theta[0] - lbda/m @ theta[0]
    return grad

X = data[:,0:2] # 100,2
y = data[:,2] # 100,
ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X)) # 100,3
y = y[:,np.newaxis] # 100,1
theta = np.zeros((3,1)) # 3,1

lbda = 0
result = opt.fmin(computeCost,x0=theta,args=(X,y,lbda),maxiter=400, full_output=True)

# J = computeCost(X,y,theta)
# print(J)


# temp = opt.fmin_tnc(func=computeCost, x0=theta.flatten(),fprime=gradient,args = (X,y.flatten()))
#
# theta_optimised = temp[0]
# print(theta_optimised)

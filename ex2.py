import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
ex2data1_path = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex2\ex2\ex2data1.txt"
data = np.loadtxt(ex2data1_path, delimiter=',')


def sigmoid(a):
    return 1/(1 + pow(np.e, -a))


def computeCost (theta,X,y,lbda):
    m = X.shape[0]
    hx = sigmoid(X @ theta)
    temp = (1/m) * (-y.T @ np.log(hx) - (1-y).T @ np.log(1-hx))
    reg = (lbda/(2*m)) * theta[1:].T @ theta[1:]
    J = temp + reg
    return J
def gradient(theta,X,y, lbda):
    m = X.shape[0]
    grad = ((1/m) * (X.T @ (sigmoid(X @ theta) - y))) + lbda/m * theta
    theta[0] = theta[0] - lbda/m @ theta[0]
    return grad


def plotData():
    plt.figure(figsize=(10, 6))
    plt.plot(pos[:, 1], pos[:, 2], 'k+', label='Admitted')
    plt.plot(neg[:, 1], neg[:, 2], 'yo', label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    boundary_xs = np.array([np.min(X[:, 1])], np.max(X[:, 1]))
    boundary_ys = (-1. / newTheta[2]) * (newTheta[0] + newTheta[1] * boundary_xs)
    plt.plot(boundary_xs, boundary_ys, 'b-', label="Decision Boundary")
    plt.legend()
    plt.grid(True)

X = data[:,0:2] # 100,2
y = data[:,2] # 100,
ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X)) # 100,3
y = y[:,np.newaxis] # 100,1
theta = np.zeros((3,1)) # 3,1
lbda = 0
print("Before optimise", computeCost(theta,X,y,lbda))
pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])
result = opt.fmin(computeCost,x0=theta,args=(X,y,lbda),maxiter=400, full_output=True)
newTheta = result[0]
print(newTheta)
print("After optimise", computeCost(newTheta,X,y,lbda))


plotData()


plt.show()


# J = computeCost(X,y,theta)
# print(J)


# temp = opt.fmin_tnc(func=computeCost, x0=theta.flatten(),fprime=gradient,args = (X,y.flatten()))
#
# theta_optimised = temp[0]
# print(theta_optimised)

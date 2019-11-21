import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

datafile = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex2\ex2\ex2data2.txt"

cols = np.loadtxt(datafile,delimiter=",",usecols=(0,1,2),unpack=True)
X = np.transpose(np.array(cols[:-1])) # 118,2
y = np.transpose(np.array(cols[-1:])) # 118,1
m = y.size

X = np.insert(X,0,1,axis=1) # 118,3

pos = np.array([X[i] for i in range (X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range (X.shape[0]) if y[i] == 0])


def plotData():
    plt.plot(pos[:, 1], pos[:, 2], "k+", label='y=1')
    plt.plot(neg[:, 1], neg[:, 2], "yo", label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.grid(True)
    plt.show()
plt.figure(figsize=(6,6))
#plotData()

def mapFeature( x1col, x2col ):
    """
    Function that takes in a column of n- x1's, a column of n- x2s, and builds
    a n- x 28-dim matrix of featuers as described in the homework assignment
    """
    degrees = 6
    out = np.ones( (x1col.shape[0], 1) )

    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 )
            out   = np.hstack(( out, term ))
    return out

mappedX = mapFeature(X[:,1],X[:,2])
def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))
def computeCost(theta, X, y, lbda):
    hx = sigmoid(X @ theta)
    temp = 1 / m * (-y.T @ np.log(hx) - (1-y).T @ np.log(1-hx))
    reg = lbda/(2*m) * theta[1:].T @ theta[1:]
    J = temp + reg
    return J

def gradient(theta, X, y, lbda):
    hx = sigmoid(X @ theta)
    temp = 1 / m * X.T @ (hx - y)
    reg = lbda/m * theta
    grad = temp + reg
    grad[0] = grad[0] - lbda/m * theta[0]
    return grad

initial_theta = np.zeros((mappedX.shape[1],1))
J = computeCost(initial_theta,mappedX,y,lbda=0)
print(J)


lbda = 0
result = opt.minimize(computeCost,initial_theta,args=(mappedX,y,lbda),method='BFGS',options={"maxiter":5000,"disp":False})
mytheta = np.array([result.x])
print(result.fun)

def optimizeRegularizedTheta(mytheta,myX,myy,mylambda=0.):
    result = opt.minimize(computeCost, mytheta, args=(myX, myy, mylambda),  method='BFGS', options={"maxiter":500, "disp":False} )
    return np.array([result.x]), result.fun

def plotBoundary(mytheta, myX, myy, mylambda=0.):
    """
    Function to plot the decision boundary for arbitrary theta, X, y, lambda value
    Inside of this function is feature mapping, and the minimization routine.
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the hypothesis classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    theta, mincost = optimizeRegularizedTheta(mytheta,myX,myy,mylambda)
    xvals = np.linspace(-1,1.5,50)
    yvals = np.linspace(-1,1.5,50)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(theta,myfeaturesij.T)
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals, [0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")

plt.figure(figsize=(12,10))
plt.subplot(221)
plotData()
plotBoundary(mytheta,mappedX,y,0.)

plt.subplot(222)
plotData()
plotBoundary(mytheta,mappedX,y,1.)

plt.subplot(223)
plotData()
plotBoundary(mytheta,mappedX,y,10.)

plt.subplot(224)
plotData()
plotBoundary(mytheta,mappedX,y,100.)
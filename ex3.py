import scipy
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt
import random
import matplotlib.cm as cm
import scipy.misc
import PIL.Image


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costFunction(theta,X, y,  lmbda):
    m = len(y)
    hx = sigmoid(X @ theta)
    reg = lmbda/(2*m) * (theta[1:] @ theta[1:].T)
    J = 1 / m * np.sum(-y * np.log(hx) - (1 - y) * np.log(1 - hx)) + reg
    return J

def gradient(theta,X, y,  lmbda):
    m = len(y)
    hx = sigmoid(X @ theta)
    reg = lmbda/m * theta
    grad = 1 / m * (X.T @ (hx - y))
    grad[0] = grad[0] - theta[0] * lmbda/m
    return grad

ex3data1_path = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex3\ex3\ex3data1.mat"

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
print ("'y' shape: %s. Unique elements in y: %s"%(data['y'].shape,np.unique(data['y'])))
print ("'X' shape: %s. X[0] shape: %s"%(X.shape,X[0].shape))

#Visualizing the data
def getDatumImg(row):
    width, height = 20,20
    square = row[1:].reshape(width,height)
    return square.T
def displayData(indices_to_display = None):
    width, height = 20, 20
    nrows, ncols = 10,10
    if not indices_to_display:
        indices_to_display = random.sample(range(X.shape[0]),nrows*ncols)
    big_picture = np.zeros((height * nrows, width * ncols))
    irow, icol = 0,0
    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = getDatumImg(X[idx])
        big_picture[irow * height : irow * height + iimg.shape[0], icol * width : icol * width + iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(6,6))
    img = PIL.Image.fromarray(big_picture)
   # plt.show(img)

displayData()

def optimizeTheta(theta, X, y, lbda=0):
    result = opt.fmin_cg(costFunction, fprime = gradient, x0 = theta, args = (X, y, lbda), maxiter = 50, disp = False, full_output= True)
    return result[0], result[1]

def buildTheta():
    lbda = 0
    initial_theta = np.zeros((X.shape[1],1)).reshape(-1)
    Theta = np.zeros((10,X.shape[1]))
    for i in range(10):
        iclass = i if i else 10
        print("Optimizing for number ", i )
        logic_Y = np.array([1 if x == iclass else 0 for x in y])
        itheta, imincost = optimizeTheta(initial_theta,X,logic_Y,lbda)
        Theta[i,:]=itheta
    return Theta
Theta = buildTheta()


def predictOneVsAll(myTheta, myrow):
    classes = np.hstack((np.array(10) , np.arange(1,10)))
    hypots = [0] * X.shape[0]
    for i in range (10):
        hypots[i] = sigmoid(myTheta[i] @ myrow)
    return classes[np.argmax(np.array(hypots))]
    # classes = [10] + range(1,10)
    # hypots = [0] * len(classes)
    # for i in range (len(classes)):
    #     hypots[i] = sigmoid(myTheta @ myrow)
    # return classes[np.argmax(np.array(hypots))]

n_correct, n_total = 0. , 0.
incorrect_indices = []
for irow in range(X.shape[0]):
    n_total += 1
    if predictOneVsAll(Theta,X[irow]) == y[irow]:
        n_correct += 1
    else: incorrect_indices.append(irow)
print("Accuracy : ", n_correct/n_total)



"""
theta = np.zeros((k,n))
for i in range (k):
    digit_class = i if i else 10
    theta[i] = opt.fmin_cg(f= costFunction,x0=theta[i],fprime=gradient,args=(X,(y == digit_class).flatten(),lmbda),maxiter = 50)

pred = np.argmax(X @ theta.T, axis = 1)
pred = [e if e else 10 for e in pred]
np.mean(pred == y.flatten()) * 100
"""

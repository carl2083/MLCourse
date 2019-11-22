import numpy as np
import scipy.io
from scipy.io import loadmat
datafile = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex3\ex3\ex3weights.mat"
mat = scipy.io.loadmat(datafile)

Theta1, Theta2 = mat['Theta1'],mat['Theta2']
print("Theta1 shape: ", Theta1.shape) # 25,401
print("Theta2 shape: ", Theta2.shape) # 10,26

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagateForward(row, Thetas):
    features = row
    for i in range (len(Thetas)):
        Theta = Thetas[i]
        z = Theta @ features
        a = sigmoid(z)
        if i == len(Thetas)-1:
            return a
        a = np.insert(a,0,1)
        features = a

def predictNN(row, Thetas):
    classes = np.arange(1,11)
    output = propagateForward(row, Thetas)
    return classes[np.argmax(np.array(output))]

ex3data1_path = r"C:\Users\chenc3\Desktop\ML\machine-learning-ex3\ex3\ex3data1.mat"

data = loadmat(ex3data1_path)
X = data['X']
y = data['y']
ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X))

myThetas = [Theta1, Theta2]
n_correct, n_total = 0.0, 0.0
incorrect_indices = []

for irow in range (X.shape[0]):
    n_total += 1
    a =  predictNN(X[irow],myThetas)
    if a == int(y[irow]):
        n_correct += 1
    else: incorrect_indices.append(irow)
print("Training accuracy: ", n_correct/n_total)

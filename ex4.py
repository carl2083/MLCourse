import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy
import itertools


datafile = r"C:\Users\carlc\Desktop\MLcourse\machine-learning-ex4\machine-learning-ex4\ex4\ex4data1.mat"
mat = scipy.io.loadmat(datafile)
X, y = mat['X'], mat['y']
X = np.insert(X,0,1,axis=1)
print(X.shape, y.shape)

thetafile = r"C:\Users\carlc\Desktop\MLcourse\machine-learning-ex4\machine-learning-ex4\ex4\ex4weights.mat"
mat = scipy.io.loadmat(thetafile)
Theta1, Theta2 = mat['Theta1'], mat['Theta2']
print(Theta1.shape, Theta2.shape)

input_layer_size = 400
hidden_layer_size = 25
output_layer_size = 10
n_training_samples = X.shape[0]

def flattenParams(thetas_list):
    flattened_list = [mytheta.flatten() for mytheta in thetas_list]
    combined = list(itertools.chain.from_iterable(flattened_list))
    assert len(combined) == (input_layer_size + 1) * hidden_layer_size + (hidden_layer_size + 1) * output_layer_size
    return np.array(combined).reshape((len(combined),1))

def reshapeParam(flattened_array):
    theta1 = flattened_array[:(input_layer_size + 1) * hidden_layer_size].reshape((hidden_layer_size,input_layer_size + 1))
    theta2 = flattened_array[(input_layer_size + 1) * hidden_layer_size:].reshape((output_layer_size, hidden_layer_size + 1))
    return [theta1, theta2]

def flattenX(myX):
    return np.array(myX.flatten()).reshape((n_training_samples * (input_layer_size+1),1))

def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((n_training_samples, input_layer_size+1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagateForward(row, Thetas):
    features = row
    zs_as_per_layer = []
    for i in range (len(Thetas)):
        Theta = Thetas[i]
        z = (Theta @ features).reshape((Theta.shape[0],1))
        a = sigmoid(z)
        zs_as_per_layer.append(((z,a)))
        if i == len(Thetas) - 1:
            return np.array(zs_as_per_layer)
        a = np.insert(a,0,1)
        features = a


def computeCost (mythetas_flattened, myX_flattened, myy, mylabda=0):
    mythetas = reshapeParam(mythetas_flattened)
    myX = reshapeX(myX_flattened)
    total_cost = 0
    m = n_training_samples

    for irow in range(m):
        myrow = myX[irow]
        myhs = propagateForward(myrow, mythetas)[-1][1]
        tempy = np.zeros((10,1))
        tempy[myy[irow]-1] = 1
        mycost = -tempy.T @ np.log(myhs) - (1-tempy.T) @ np.log(1-myhs)
        total_cost += mycost
    total_cost = float(total_cost)/m
    total_reg = 0.
    for mytheta in mythetas:
        total_reg += np.sum(mytheta* mytheta)
    total_reg *= float(mylabda)/(2*m)

    return total_cost + total_reg

myThetas = [Theta1, Theta2]

print ("Cost:", computeCost(flattenParams(myThetas),flattenX(X),y))

def sigmoidGradient(z):
    dummy = sigmoid(z)
    return dummy * (1-dummy)

def genRandThetas:
    epsilon_init = 0.12
    theta1_shape = (hidden_layer_size, input_layer_size + 1)
    theta2_shape = (output_layer_size, hidden_layer_size + 1)
    rand_thetas = [np.random.rand( *theta1_shape) * 2 * epsilon_init - epsilon_init, np.random.rand( *theta2_shape) * 2 * epsilon_init - epsilon_init]
    return rand_thetas

def backPropagate(mythetas_flattened, myX_flattened, myy, mylambda=0.):
    mythetas = reshapeParam(mythetas_flattened)
    myX = reshapeX(myX_flattened)

    Delta1 = np.zeros((hidden_layer_size, input_layer_size + 1))
    Delta2 = np.zeros((output_layer_size, hidden_layer_size + 1))
    m = n_training_samples
    for irow in range(m):
        myrow = myX[irow]
        a1 = myrow.reshape((input_layer_size,1))
        temp = propagateForward(myrow, mythetas)
        t2 = temp[0][0]
        a2 = temp[0][1]
        z3 = temp[1][0]
        a3 = temp[1][1]
        tempy = np.zeros((10,1))
        tempy[myy[irow]-1] = 1
        delta3 = a3 - tempy
        delta2 = mythetas[1].T[1:,:] @ delta3 * sigmoidGradient(z2)
        a2 = np.insert(a2,0,1,axis=0)
        Delta1 += delta2 @ a1.T
        Delta2 += delta3 @ a2.T
    D1 = Delta1/float(m)
    D2 = Delta2/float(m)
    D1[:,1:] = D1[:,1:] + (float(mylambda)/m)*mythetas[0][:,1:]
    D2[:,1:] = D2[:,1:] + (float(mylambda)/m)*mythetas[1][:,1:]
    
    return flattenParams([D1,D2]).flatten()

flattenedD1D2 = backPropagate(flattenParams(myThetas),flattenX(X),y,mylambda=0.)

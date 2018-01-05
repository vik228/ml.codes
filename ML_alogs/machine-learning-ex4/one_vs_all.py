import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost_reg(theta, X, y, reg_param):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply(1-y, np.log(1 - sigmoid(X*theta.T)))
    reg = (reg_param/2 * len(X))*np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second)/(len(X)) + reg

def gradient(theta, x, y, reg_param):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    error = sigmoid(x*theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((reg_param / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    return np.array(grad).ravel()

def one_vs_all(x, y, num_labels, reg_param):
    rows = x.shape[0]
    params = x.shape[1]
    all_theta = np.zeros((num_labels, params+1))
    x = np.insert(x, 0, values=np.ones(rows), axis=1)
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        fmin = minimize(fun=cost_reg, x0=theta, args=(x, y_i, reg_param), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x
    return all_theta



if __name__== "__main__":
    data = loadmat('./ex3data1.mat')
    rows = data['X'].shape[0]
    params = data['X'].shape[1]

    all_theta = np.zeros((10, params + 1))

    X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

    theta = np.zeros(params + 1)

    y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
    y_0 = np.reshape(y_0, (rows, 1))

    print (X.shape, y_0.shape, theta.shape, all_theta.shape)
    all_theta = one_vs_all(data['X'], data['y'], 10, 1)
    print (all_theta)


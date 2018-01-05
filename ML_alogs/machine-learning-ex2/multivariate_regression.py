import os
import numpy as np
import pandas as pd

def compute_cost(X, y, theta):
    inner = np.power(((X*theta.T) - y), 2)
    return np.sum(inner)/(2*len(X))


def compute_gradient(X, y, theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X*theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = temp[0, j] - ((alpha/len(X)))*np.sum(term)
        theta = temp
        cost[i] = compute_cost(X, y, theta)
    return theta, cost



if __name__ == "__main__":
    path = "{base_dir}/ex1data2.txt".format(base_dir=os.getcwd());
    data = pd.read_csv(path, header=None,names=['Size', 'Bedroom', 'Price'])
    data = (data - data.mean())/data.std()

    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0,0,0]))
    g, cost = compute_gradient(X, y, theta, 0.01, 1000)
    print compute_cost(X, y, g)


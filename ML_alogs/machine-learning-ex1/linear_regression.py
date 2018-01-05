import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def examine_data(data):
    print data.head()
    print data.describe()
    data.plot(kind='scatter', x='population', y='profit', figsize=(12,8))
    plt.show()

def compute_cost(X, y, theta):
    inner = np.power((X*theta.T - y), 2)
    return np.sum(inner)/(2*len(X))

def gradient_descent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha/len(X)))*np.sum(term)
        theta = temp;
    return theta, cost

def plot_result(data, g):
    x = np.linspace(data.population.min(), data.population.max(), 100)
    f = g[0, 0] + (g[0,1]*x)
    fig,ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Predection')
    ax.scatter(data.population, data.profit, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted profit vs population Size')
    plt.show()

if __name__== "__main__":
    path = "{base_path}/ex1data1.txt".format(base_path=os.getcwd())
    data = pd.read_csv(path, header=None, names=['population', 'profit'])
    examine_data(data)
    # append the ones column to the front of the data set
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # convert the dataframe matrices into numpy matrices
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0,0]))

    print X.shape, theta.shape, y.shape
    print compute_cost(X, y, theta)

    # initialize variables for learning rates and iterations
    alpha = 0.01
    iters = 1000

    g, cost = gradient_descent(X, y, theta, alpha, iters)

    print compute_cost(X, y, g)
    plot_result(data, g)

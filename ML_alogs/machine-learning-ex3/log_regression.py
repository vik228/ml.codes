import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt


def plot_data(data):
	positive = data[data['Admitted'].isin([1])]
	negative = data[data['Admitted'].isin([0])]

	fig, ax = plt.subplots(figsize=(12, 8))
	ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50,c='b', marker='o', label='Admitted')
	ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50,c='r',marker='x', label='Not Admitted')
	ax.legend()
	ax.set_xlabel('Exam 1 Score')
	ax.set_ylabel('Exam 2 Score')
	plt.show()

def plot_sigmoid():
	nums = np.arange(-10, 10, step=1)
	fix, ax = plt.subplots(figsize=(12, 8))
	ax.plot(nums, sigmoid(nums), 'r')
	plt.show()

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cost(theta, X, Y):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(Y)
	first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
	second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
	return np.sum(first - second)/len(X)

def gradient(theta, X, y):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)

	parameters = int(theta.ravel().shape[1])
	grad = np.zeros(parameters)

	error = sigmoid(X*theta.T) - y
	for i in range(parameters):
		term = np.multiply(error, X[:, i])
		grad[i] = np.sum(term) / len(X)
	
	return grad

def predict(theta, X):
	print theta
	print X
	probability = sigmoid(X*theta.T)
	return [1 if x >= 0.5 else 0 for x in probability]

if __name__ == "__main__":
	path = os.getcwd() + '/ex2data1.txt'
	data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
	data.insert(0, 'Ones', 1)
	cols = data.shape[1]
	X = data.iloc[:, 0:cols-1]
	y = data.iloc[:, cols-1:cols]
	X = np.array(X.values)
	y = np.array(y.values)
	theta = np.zeros(3)
	print(X.shape, y.shape, theta)
	result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
	theta_min = np.matrix(result[0])
	predictions = predict(theta_min, X)
	correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
	accuracy = (sum(map(int, correct)) % len(correct))  
	print 'accuracy = {0}%'.format(accuracy)

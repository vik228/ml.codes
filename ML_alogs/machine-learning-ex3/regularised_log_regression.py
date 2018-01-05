import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt

def plot_data(data2):
	positive = data2[data2['Accepted'].isin([1])]
	negative = data2[data2['Accepted'].isin([0])]
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o',label='Accepted')
	ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
	ax.legend()
	ax.set_xlabel('Test 1 Score')
	ax.set_ylabel('Test 2 Score')
	plt.show()

def sigmoid(z):
	return 1/(1+np.exp(-z))

def cost_reg(theta, X, y, reg_param):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
	second = np.multiply(1-y, np.log(1 - sigmoid(X*theta.T)))
	reg = (reg_param/2 * len(X))*np.sum(np.power(theta[:,1:theta.shape[1]], 2))
	return np.sum(first - second)/(len(X)) + reg

def modify_data(data2, degree):
	"""
		This method generated all possible combinations of features with a particular degree
	"""
	x1 = data2['Test 1']
	x2 = data2['Test 2']
	data2.insert(3, 'Ones', 1)
	for i in range(1, degree):
		for j in range(0, i):
			data2['F' + str(i) + str(j)] = np.power(x1, i-j)*np.power(x2, j)

	data2.drop('Test 1', axis=1, inplace=True)
	data2.drop('Test 2', axis=1, inplace=True)

def gradient_reg(theta, X, y, reg_param):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	grad = np.zeros(theta.shape[1])
	error = sigmoid(X*theta.T) - y
	for i in range(theta.shape[1]):
		term = np.multiply(error, X[:, i])
		if (i == 0):
			grad[i] = np.sum(term)/len(X)
		else:
			grad[i] = (np.sum(term)/len(X)) + ((reg_param/len(X))*theta[:, i])
	return grad

def predict(theta, X):
	probablity = sigmoid(X*theta.T)
	return [1 if x >= 0.5 else 0 for x in probablity]

if __name__== "__main__":
	file_path = os.getcwd() + '/ex2data2.txt'
	data2 = pd.read_csv(file_path, header=None,names=["Test 1", "Test 2", "Accepted"])
	modify_data(data2, 5)
	cols = data2.shape[1]
	x2 = data2.iloc[:, 1:cols]
	y2 = data2.iloc[:, 0:1]
	x2 = np.array(x2.values)
	y2 = np.array(y2.values)
	theta2 = np.zeros(x2.shape[1])
	result2 = opt.fmin_tnc(func=cost_reg, x0=theta2, fprime=gradient_reg, args=(x2, y2,1))
	theta_min = np.matrix(result2[0])
	predictions = predict(theta_min,x2)
	correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
	accuracy = (sum(map(int, correct)) % len(correct))
	print 'accuracy = {0}%'.format(accuracy)



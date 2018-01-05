import numpy as np

def gradientDescent(x, y, theta, alpha, m, iterations):
	xTrans = x.transpose()
	for i in range(0, iterations):
		hypothesis = np.dot(x, theta)
		loss = hypothesis - y
		gradient = np.dot(xTrans, loss)/m
		theta = theta - alpha*gradient
	return theta

if __name__ == "__main__":
	data = []
	for i in range(0,84):
		data.append([np.float(x) for x in raw_input().split(',')])
	np_array = np.array(data)

	X = np_array[:, 0:1]
	Y = np_array[:, 1:]
	Y = Y.ravel()
	X = np.insert(X, 0, np.ones(1), 1)
	theta = np.ones(2)
	alpha = 0.01
	iterations = 1500
	theta = gradientDescent(X, Y, theta ,alpha,84 , iterations)
	print(theta)


import numpy as np
def gradentDescent(x, y, theta, alpha, m, iterations):
	xTrans = x.transpose()
	for i in range(0, iterations):
		hypothesis = np.dot(x, theta)
		loss = hypothesis - y
		gradient = np.dot(xTrans, loss)/m
		theta = theta - alpha*gradient
	return theta

arr = [int(x) for x in raw_input().split()]
n = arr[0]
m = arr[1]
data = []
for i in range(0, m):
	data.append([np.float(x) for x in raw_input().split()])
np_array = np.array(data)
X = np_array[:, 0:n]
Y = np_array[:, n:]
Y = Y.ravel()
X = np.insert(X, 0, np.ones(1), 1)
theta = np.ones(n+1)
print X
print theta
alpha = 0.005
iterations = 100000
theta = gradentDescent(X, Y, theta, alpha, m, iterations)
t = int(raw_input())
data1 = []
for i in range(0, t):
	data1.append([np.float(x) for x in raw_input().split()])

np_array1 = np.array(data1)
x = np.insert(np_array1, 0,np.ones(1), 1)
ans = np.dot(x, theta)
for i in range(0, t):
	print("%.2f" % ans[i])


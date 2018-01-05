import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

def sigmoid(z):
    return 1/(1 + np.exp(-z))

if __name__ == "__main__":
    data = loadmat('./ex3data1.mat')
    X = data['X']
    y = data['y']
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)



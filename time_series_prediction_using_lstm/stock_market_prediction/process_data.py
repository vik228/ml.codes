import os
import time
import warnings
import numpy as np

class ProcessData:


    def split_data(self,result):
        row = round(0.9*result.shape[0])
        train = result[:int(row), :]
        np.random.shuffle(train)
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1] 
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_train, y_train, x_test, y_test
    

    def normalise_windows(self,results):
        normalised_data = []
        for result in results:
            try:
                normalised_data.append([((float(x)/float(result[0])) - 1) for x in result])
            except Exception as e:
                print result
        return normalised_data
    
    
    def load_data(self,filename, seq_len, normalise_window):
        f = open(filename,'rb').read()
        data = f.decode().split("\n")
        sequence_length = seq_len + 1
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index:index+sequence_length])
        if normalise_window:
            result = self.normalise_windows(result)

        result = np.array(result)
        return self.split_data(result)









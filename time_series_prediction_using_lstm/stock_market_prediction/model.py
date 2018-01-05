from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from process_data import ProcessData
import time
import numpy as np
from numpy import newaxis

class Model:

    model = None
    
    def __init__(self):
        self.model = Sequential()
        self.build_model([1,50,100,1])

    def build_model(self,layers):
        self.model.add(LSTM(
            input_shape=(layers[1], layers[0]),
            output_dim=layers[1],
            return_sequences=True))
        self.model.add(Dropout(0.2))
        
        self.model.add(LSTM(
        layers[2],
        return_sequences=False))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(
            output_dim=layers[3]))
        self.model.add(Activation("linear"))

        start = time.time()
        self.model.compile(loss="mse", optimizer="rmsprop")
        print("> Compilation Time : ", time.time() - start)

    def predict_point_by_point(self,model, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predicted = model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequence_full(self,model, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        return predicted

    def predict_sequence_multiple(self,model, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs
    
    def init_lstm(self,filename, seq_len, epochs):
        data_processor = ProcessData()
        x_train, y_train, x_test, y_test = data_processor.load_data(filename, seq_len, True)
        self.model.fit(x_train,y_train,batch_size=512,epochs=epochs,validation_split=0.05)
        predictions = self.predict_sequence_multiple(self.model, x_test, seq_len, 50)
        return predictions,y_test
        



    

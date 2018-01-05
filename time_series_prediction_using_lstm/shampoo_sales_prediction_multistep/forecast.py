from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt

class Forecast:
    
    def persistence(self,last_ob, n_seq):
        return [last_ob for i in range(n_seq)]

    def make_forecasts(self,train, test, n_lag, n_seq):
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            forecast = self.persistence(X[-1], n_seq)
            forecasts.append(forecast)
        return forecasts

    def evaluate_forecasts(self,test, forecasts, n_lag, n_seq):
        for i in range(n_seq):
            actual = test[:,(n_lag + i)]
            predicted = [forecast[i] for forecast in forecasts]
           
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i+1), rmse))


    def plot_forecasts(self,series, forecasts, n_test):
        plt.plot(series.values)
        for i in range(len(forecasts)):
            off_s = len(series) - 12 + i - 1
            off_e = off_s + len(forecasts[i]) + 1
            xaxis = [x for x in range(off_s, off_e)]
            yaxis = [series.values[off_s]] + forecasts[i]
            plt.plot(xaxis, yaxis, color='red')
        plt.show()



   

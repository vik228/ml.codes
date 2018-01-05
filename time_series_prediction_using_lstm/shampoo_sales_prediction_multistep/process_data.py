import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from pandas import datetime
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
def parser(x):
    return datetime.strptime("190"+x, '%Y-%m')


class ProcessData:

    def series_to_supervised(self,data, n_in=1, n_out=1,dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    def difference(dataset, interval=1):
        diff = list()
        for in in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)
    
    def get_stationary_series(self,series):
        diff_series = difference(raw_values, 1)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        return diff_values
    

    def get_scaled_values(diff_series):
        scaler = MinMaxScaler(feature_range=(-1,1))
        scalled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        return scaler,scaled_values

    
    def prepare_data(self,series, n_test,n_lag,n_seq):
        raw_values = series.values
        diff_series = self.get_stationary_series(raw_values)
        scaler,scaled_values = self.get_scaled_values(diff_series)
        supervised = self.series_to_supervised(scaled_values, n_lag, n_seq)
        supervised_values = supervised.values
        train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
        return scaler,train, test



    def read_data(self,filename):
        series = pd.read_csv(filename, header=0, parse_dates=[0],index_col=0,squeeze=True, date_parser=parser)
        scaler,train, test = self.prepare_data(series, 10,1,3)
        return series,scaler,train, test
        


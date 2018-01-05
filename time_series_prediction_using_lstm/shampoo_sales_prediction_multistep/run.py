from process_data import ProcessData
from forecast import Forecast

def main():
    data_processor = ProcessData()
    forecast_util = Forecast()
    series,train, test = data_processor.read_data('shampoo-sales.csv')

    forecasts = forecast_util.make_forecasts(train,test, 1, 3)
    forecast_util.evaluate_forecasts(test, forecasts, 1, 3)
    forecast_util.plot_forecasts(series, forecasts, 12)

if __name__ == '__main__':
    main()

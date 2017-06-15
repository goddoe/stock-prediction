from pandas_datareader import data as web
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pickle

class StockPredictor(object):

    def __init__(self, stock_dict_path):

        delta = relativedelta(months=12)

        self.start_date = datetime.now()-delta
        self.end_date = datetime.now()

        with open(stock_dict_path, "rb") as f:
            self.stock_dict = pickle.load(f)


    def predict(self, query):
        stock_data = self.crawl_data(query, self.start_date, self.end_date)
        high_price_predictor = self.generate_model(stock_data, 'Low')
        low_price_predictor = self.generate_model(stock_data, 'High')

        high_price = high_price_predictor['model'].predict(high_price_predictor['query_x'])
        low_price = low_price_predictor['model'].predict(low_price_predictor['query_x'])
        
        result = {
                    'high_price':high_price[0],
                    'low_price':low_price[0],
                    'high_price_predictor':high_price_predictor,
                    'low_price_predictor': low_price_predictor
                }

        return result


    def crawl_data(self, query, start_date, end_date):

        stock =  self.stock_dict['items'][query]

        ticker =stock['ticker']
        market = stock['market']
        if market == 'kospi':
            market = 'KS'
        elif market == 'kosdaq':
            market = 'KQ'


        stock_code = ticker+'.'+market
        stock_data = web.DataReader(stock_code, "yahoo", start_date, end_date)

        return stock_data


    def generate_model(self, stock_data, y_type='Open', kernel_type='linear'):
        model = SVR(kernel=kernel_type, C=1e3)

        feature_dimension = 6
        x = np.ndarray(shape=(len(stock_data['Open']), feature_dimension), dtype='float32')

        feature_tmp = [stock_data['Open']
            , stock_data['High']
            , stock_data['Low']
            , stock_data['Close']
            , stock_data['Volume']
            , stock_data['Adj Close']] 

        for i, feature in enumerate(feature_tmp):
            feature2d = np.array([list(feature)])
            feature2d = np.rot90(feature2d, -1)
            
            mean = np.sum(feature2d)/len(feature2d)
            variance = np.sum((feature2d-mean)*(feature2d-mean))/ (len(feature2d)-1)
            standard_deviation = np.sqrt(variance)
            
            feature2d = (feature2d - mean)/standard_deviation
            
            x[:,i] = feature2d[:,0]

        #Erase last day
        query_x = x[-1:,:]
        x = x[:-1,:]


        y = np.array(list(stock_data[y_type]), dtype='float32')
        y_origin = y.copy()

        y_mean = np.sum(y)/len(y)
        variance = np.sum((y-y_mean)*(y-y_mean))/ (len(y)-1)
        standard_deviation = np.sqrt(variance)

        y = (y-y_mean) / standard_deviation
        #Erase First day
        y = y[1:]

        y_lin = []

        train_data_window = 50
        for i in range(len(y) - train_data_window ):
            y_lin.append(model.fit(x[i:train_data_window+i],y[i:train_data_window+i]).predict(np.array([x[train_data_window+i]]))[0])
        days = np.array(stock_data.index)
        days = days[train_data_window+1:]

        result ={
                    'model':model,
                    'query_x':query_x,
                    'predicted_pre_y': y_lin*standard_deviation + y_mean,
                    'origin_y': y_origin[train_data_window:] ,
                    'days' : days 
                }

        return result

def main():

    # read stock data
    stock_predictor = StockPredictor('./stock_dict_from_db.pickle')
    result = stock_predictor.predict('삼성전자')
    print(result)

    plt.hold('on')
    plt.plot(result['days'],y['origin_y'],c='k', label='day')
    plt.plot(result['days'], y['predicted_pre_y'], c='r', label='Linear model')
    #plt.plot(x, y_poly, c='b', label='Poly model')

    plt.xlabel('days')
    plt.ylabel('Normalized Open Price')
    plt.title('Stock Prediction using SVR')
    plt.legend()
    plt.show()



if __name__=='__main__':
    main()

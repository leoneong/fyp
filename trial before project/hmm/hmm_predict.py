import numpy as np
import investpy
import talib
import pandas as pd
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import GaussianHMM

def get_stock_data(a):
    df = investpy.get_stock_historical_data(stock= a,
                                        country='malaysia',
                                        from_date='01/01/2019',
                                        to_date='30/05/2020')
    return df

def get_stock_data_year(a,date1,date2):
    df = investpy.get_stock_historical_data(stock= a,
                                        country='malaysia',
                                        from_date= date1,
                                        to_date= date2)
    return df

def get_company_profile(a):
    company_profile = investpy.get_stock_company_profile(stock=a,
                                                     country='malaysia')
    return company_profile.get('desc','')

def get_feature(a,n):
    #1. data gathering & processing
    data = get_stock_data(a)
    data = data.replace(0, np.nan)
    data.dropna(inplace=True)

    #2. exponential smoothing
    S = X = np.array(data['Close'])
    alpha = 0.9
    for i in range(1,len(S)):
        S[i] = alpha*X[i] + (1-alpha)*S[i-1]
    data['Close'] = S

    
    #3.data extraction
    macd, dea, bar = talib.MACD(data['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    fastk, fastd = talib.STOCHF(data['High'], data['Low'], data['Close'], fastk_period=14, fastd_period=3, fastd_matype=0)
    data['dif'] = data['Close'].diff(-n)
    data['MACD'] = macd
    data['STOCH'] = fastk
    data['WILLR'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['ATR'] = talib.ATR(data['High'],data['Low'],data['Close'], timeperiod=14)
    data = pd.DataFrame(data)
    data.dropna(inplace=True)
    

    #4. Labels are the values we want to predict
    diff = np.array(data['dif'])
    labels = np.zeros(len(diff))
    sum = 0
    for i in range(len(diff)):
        if diff[i] > 0:
            labels[i] = 1
            sum += 1
    
    # print (sum/len(diff))


    #5. Remove the excess data from the features
    features = data[['MACD','STOCH','WILLR','OBV','ATR','RSI']]

    #6. Saving feature names for later use
    feature_list = list(features.columns)
    return features, labels, feature_list

def get_feature_year(a,n,date1,date2):
    #1. data gathering & processing
    data = get_stock_data_year(a,date1,date2)
    data = data.replace(0, np.nan)
    data.dropna(inplace=True)

    #2. exponential smoothing
    S = X = np.array(data['Close'])
    alpha = 0.9
    for i in range(1,len(S)):
        S[i] = alpha*X[i] + (1-alpha)*S[i-1]
    data['Close'] = S

    
    #3.data extraction
    macd, dea, bar = talib.MACD(data['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    fastk, fastd = talib.STOCHF(data['High'], data['Low'], data['Close'], fastk_period=14, fastd_period=3, fastd_matype=0)
    data['dif'] = data['Close'].diff(-n)
    data['MACD'] = macd
    data['STOCH'] = fastk
    data['WILLR'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['ATR'] = talib.ATR(data['High'],data['Low'],data['Close'], timeperiod=14)
    data = pd.DataFrame(data)
    data.dropna(inplace=True)
    

    #4. Labels are the values we want to predict
    diff = np.array(data['dif'])
    labels = np.zeros(len(diff))
    sum = 0
    for i in range(len(diff)):
        if diff[i] > 0:
            labels[i] = 1
            sum += 1
    
    # print (sum/len(diff))


    #5. Remove the excess data from the features
    features = data[['STOCH']]

    #6. Saving feature names for later use
    feature_list = list(features.columns)
    return features, labels, feature_list

def predict_hmm(features, labels, feature_list):
    #7. train and fit
    # Convert to numpy array
    features = np.array(features)
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features,labels, test_size = 0.25,shuffle=False)
    # Instantiate model 
    hmm = GaussianHMM(n_components=3)
    # Train the model on training data
    hmm = hmm.fit(train_features)
    # Use the forest's predict method on the test data
    predictions = hmm.predict(test_features)
 
    return predictions[-1], predictions, test_labels

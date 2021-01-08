import numpy as np
import investpy
import talib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def get_stock_data(a, start_date, end_date):
    df = investpy.get_stock_historical_data(stock= a,
                                        country='malaysia',
                                        from_date=start_date,
                                        to_date=end_date)
    return df

def get_feature(a, start_date, end_date):
    #1. data gathering & processing
    data = get_stock_data(a, start_date, end_date)
    
    data.reset_index(inplace=True)
    data = data.replace(0, np.nan)
    data.dropna(inplace=True)

    #3.data extraction
    macd, dea, bar = talib.MACD(data['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    fastk, fastd = talib.STOCHF(data['High'], data['Low'], data['Close'], fastk_period=14, fastd_period=3, fastd_matype=0)
    data['dif'] = data['Close'].diff()/data['Close'].shift(1)
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
    price = np.array(data['Close'])
 
    labels = np.zeros(len(diff))
  
    for i in range(len(diff)):
        if diff[i] > 0:
            labels[i] = 1
            

    #5. Data for the features
    features = data[['Date','MACD','STOCH','WILLR','OBV','ATR','RSI']]

    #6. Saving feature names for later use
    feature_list = list(features.columns)
    return features, labels, diff

# extract TA data
train_x, train_y, diff_train = get_feature('IOIB','01/01/2015','31/12/2019')
test_x, test_y, diff_test = get_feature('IOIB','01/01/2020','30/09/2020')

# raed FA data
df = pd.read_excel('data/stock_fundamental_data.xlsx', sheet_name='IOICORP')

# format date
df['Date'] = pd.to_datetime(df.Date, format= '%Y-%m-%d')

# fill nan value for revenue and profit back filling
df = df[['Date','Revenue','Net Profit']]
merge_df = pd.merge(train_x,df,on='Date',how='outer')
merge_df = merge_df.fillna(method='ffill')

# combine TA indicator with FA indicator
merge_df = merge_df[['Date','Revenue','Net Profit']]
train_x['label'] = train_y 
merge_df = pd.merge(train_x,merge_df,on='Date',how='left')
merge_df = merge_df.dropna()
merge_df = merge_df[:-1]

# separate label
train_y = merge_df['label']
merge_df = merge_df.drop(columns =['label','Date'])
print(merge_df)

from sklearn.model_selection import RandomizedSearchCV

param_grid = [
    {'max_depth':[3,4,5,6,7,8,9,10],
    'max_features':[2,3,4,5],
    'n_estimators':[100,200,300,400,500]},    
]

rf_clf = RandomForestClassifier()
random_search = RandomizedSearchCV(rf_clf, param_grid, cv = 4, scoring='average_precision', return_train_score= True)
random_search.fit(merge_df,train_y)



print(random_search.best_estimator_)
print(random_search.best_estimator_.feature_importances_)

from sklearn.metrics import confusion_matrix, precision_score, recall_score
test_x = test_x.drop(columns=['Date'])
y_random_pred = random_search.predict(merge_df)
precision = precision_score(train_y, y_random_pred)
print(precision)

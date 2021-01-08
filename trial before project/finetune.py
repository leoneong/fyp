import numpy as np
import investpy
import talib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def get_stock_data(a, start_date, end_date):
    df = investpy.get_stock_historical_data(stock= a,
                                        country='malaysia',
                                        from_date=start_date,
                                        to_date=end_date)
    return df

def get_feature(a, start_date, end_date):
    #1. data gathering & processing
    data = get_stock_data(a, start_date, end_date)
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
    features = data[['MACD','STOCH','WILLR','OBV','ATR','RSI']]

    #6. Saving feature names for later use
    feature_list = list(features.columns)
    return features, labels, diff


train_x, train_y, diff_train = get_feature('IOIB','01/01/2015','31/12/2019')
test_x, test_y, diff_test = get_feature('IOIB','01/01/2020','30/09/2020')

tree_clf = DecisionTreeClassifier(max_depth = 6)
tree_clf.fit(train_x,train_y)

from sklearn.model_selection import cross_val_score
y_train_accuracy = cross_val_score(tree_clf, train_x, train_y, cv = 4, scoring = "accuracy")

from sklearn.model_selection import cross_val_predict
y_train_pred= cross_val_predict(tree_clf, train_x, train_y, cv = 4)

# confusion matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score
cm = confusion_matrix(train_y, y_train_pred)
# print(cm) 
precision = precision_score(train_y, y_train_pred)
recall = recall_score(train_y, y_train_pred)

# output predict probability
y_train_pred_proba= cross_val_predict(tree_clf, train_x, train_y, cv = 4, method="predict_proba")
y_score = y_train_pred_proba[:,1]

# increase treshold
y_adjusted_pred =  np.zeros(len(y_score))
for i in range(len(y_score)):
    if (y_score[i]) > 0.6:
        y_adjusted_pred[i] = 1


precision1 = precision_score(train_y, y_adjusted_pred)
# print(precision)
# print(precision1)
# cm1 = confusion_matrix(train_y, y_adjusted_pred)
# print(cm1) 

def backtest(predictions, diff):
    period = len(diff)- len(predictions)
    sReturn= diff[period:]

    #cost of each trade
    Cost=pd.Series(np.zeros(len(predictions))) 
    for i in range(1,len(predictions)):
        if predictions[i-1]!=predictions[i]:
            Cost.values[i]=0.00042
    SignalRe=np.cumprod(predictions*(sReturn-Cost)+1)-1 #strategy cumulative return
    IndexRe=np.cumprod(sReturn+1)-1 #index cummulative return
    SignalRe = np.array(SignalRe)
    IndexRe = np.array(IndexRe)
    return SignalRe[-1]

# print(backtest(y_train_pred,diff_train))
# print(backtest(y_adjusted_pred,diff_train))


# grid search
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'max_depth':[3,4,5,6],
    'max_features':[2,3,4,5],
    },
    
]
tree_clf1 = DecisionTreeClassifier()


grid_search = GridSearchCV(tree_clf1, param_grid, cv = 4, scoring='accuracy', return_train_score= True)
grid_search.fit(train_x,train_y)

from joblib import dump, load
# save the model to disk
filename = 'finalized_model.joblib'
dump(grid_search, filename)

grid_search1 = load(filename)

print(grid_search1.best_estimator_)


y_grid_pred = grid_search1.predict(test_x)
precision = precision_score(test_y, y_grid_pred)
print(precision)

tree_clf1.fit(train_x,train_y)
y_tree_pred = tree_clf1.predict(test_x)
precision1 = precision_score(test_y, y_tree_pred)
print(precision1)
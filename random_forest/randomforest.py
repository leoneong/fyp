import datetime
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from sklearn.preprocessing import scale
import talib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

#1. data gathering & processing
features = pd.read_csv('data/SIME.csv')
features = features.replace(0, np.nan)
features.dropna(inplace=True)
features['Date'] = pd.to_datetime(features.Date, format= '%Y-%m-%d')

#2. exponential smoothing
S = X = np.array(features['Close'])
alpha = 0.9
for i in range(1,len(S)):
    S[i] = alpha*X[i] + (1-alpha)*S[i-1]
features['Close'] = S

#3.features extraction
macd, dea, bar = talib.MACD(features['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
fastk, fastd = talib.STOCHF(features['High'], features['Low'], features['Close'], fastk_period=14, fastd_period=3, fastd_matype=0)
real = talib.WILLR(features['High'], features['Low'], features['Close'], timeperiod=14)
features['dif'] = features['Close'].diff(-30)
features['MACD'] = macd
features['STOCH'] = fastk
features['WILLR'] = real
features['OBV'] = talib.OBV(features['Close'], features['Volume'])
features['RSI'] = talib.RSI(features['Close'], timeperiod=14)
features['ATR'] = talib.ATR(features['High'],features['Low'],features['Close'], timeperiod=14)
features.dropna(inplace=True)
# print('The shape of our features is:', features.shape)

#4. Labels are the values we want to predict
diff = np.array(features['dif'])
labels = np.zeros(len(diff))
for i in range(len(diff)):
    if diff[i] > 0:
        labels[i] = 1

#5. Remove the excess data from the features
features= features.drop('Close', axis = 1)
features= features.drop('dif', axis = 1)
features= features.drop('Date', axis = 1)
features= features.drop('High', axis = 1)
features= features.drop('Low', axis = 1)
features= features.drop('Open', axis = 1)
features= features.drop('Adj Close', axis = 1)
features= features.drop('Volume', axis = 1)
# print('The shape of our features is:', features.shape)

#6. Saving feature names for later use
feature_list = list(features.columns)
 
#7. train and fit
# Convert to numpy array
features = np.array(features)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features,labels, test_size = 0.25,shuffle=False)
# Instantiate model 
rf = RandomForestClassifier(n_estimators=1000)
# Train the model on training data
rf = rf.fit(train_features, train_labels)
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)



# decision tree

# clf = tree.DecisionTreeClassifier() # 引入模型
# result = clf.fit(train_features,train_labels) # 训练模型
# predictions = clf.predict(test_features)





#8. list importance feature
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

#9. Get accuracy, precision, recall and specificity
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
for i in range(len(test_labels)):
    if predictions[i] == 1 and test_labels[i] == 1:
        true_positive += 1
    elif predictions[i] == 1 and test_labels[i] == 0:
        false_positive += 1
    elif predictions[i] == 0 and test_labels[i] == 0:
        true_negative += 1
    elif predictions[i] == 0 and test_labels[i] == 0:
        false_negative += 1

accuracy = (true_positive + true_negative)/(len(test_labels))
precision = true_positive/(true_positive + false_positive)
recall = true_positive/(true_positive + false_negative)
specificity = true_negative/(true_negative + false_positive)

print(false_negative)
print("Accuracy = " , accuracy)
print("Precision = " , precision)
print("Recall = " , recall)
print("Specificity = " , specificity)

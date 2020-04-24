import datetime
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from sklearn.preprocessing import scale
import talib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



features = pd.read_csv('data/SIME.csv')
features = features.replace(0, np.nan)
features.dropna(inplace=True)
features['Date'] = pd.to_datetime(features.Date, format= '%Y-%m-%d')

#features extraction
dif, dea, bar = talib.MACD(features['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
features['dif'] = features['Close'].diff(-1)
features['MACD'] = bar
features['RSI'] = talib.RSI(features['Close'], timeperiod=14)
features['ATR'] = talib.ATR(features['High'],features['Low'],features['Close'], timeperiod=14)
features.dropna(inplace=True)
dates = features['Date']
print('The shape of our features is:', features.shape)

# Labels are the values we want to predict
# 1 for up, 0 for down
diff = np.array(features['dif'])
labels = np.zeros(len(features))
for i in range(len(features)):
    if diff[i] > 0.0:
        labels[i] = 1

# Remove the useless data from the features
# axis 1 refers to the columns
features= features.drop('dif', axis = 1)
features= features.drop('Date', axis = 1)
features= features.drop('High', axis = 1)
features= features.drop('Low', axis = 1)
features= features.drop('Open', axis = 1)
features= features.drop('Adj Close', axis = 1)
print('The shape of our features is:', features.shape)

# Saving feature names for later use
feature_list = list(features.columns)
 
# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features,labels, test_size = 0.25,shuffle=False)

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)
 
# Train the model on training data
result = rf.fit(train_features, train_labels)

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

print(result)
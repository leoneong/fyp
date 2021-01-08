import numpy as np
import investpy
import talib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, plot_roc_curve
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint


def get_stock_data(a, start_date, end_date):
    df = investpy.get_stock_historical_data(stock=a,
                                            country='malaysia',
                                            from_date=start_date,
                                            to_date=end_date)
    return df


def get_feature(a, start_date, end_date):
    # 1. data gathering & processing
    data = get_stock_data(a, start_date, end_date)

    data.reset_index(inplace=True)
    data = data.replace(0, np.nan)
    data.dropna(inplace=True)

    # 3.data extraction
    macd, dea, bar = talib.MACD(
        data['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    fastk, fastd = talib.STOCHF(
        data['High'], data['Low'], data['Close'], fastk_period=14, fastd_period=3, fastd_matype=0)
    data['dif'] = data['Close'].diff()/data['Close'].shift(1)
    data['MACD'] = macd
    data['STOCH'] = fastd
    data['WILLR'] = talib.WILLR(
        data['High'], data['Low'], data['Close'], timeperiod=14)
    # data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    # data['ATR'] = talib.ATR(data['High'], data['Low'],
    #                         data['Close'], timeperiod=14)
    # data['MOM'] = talib.MOM(data['Close'], timeperiod=10)
    # data['MA20'] = talib.MA(data['Close'], timeperiod=20, matype=0)
    # data['MA60'] = talib.MA(data['Close'], timeperiod=60, matype=0)
    # data['Different_MA20'] = (data['MA20'] - data['Close'])/data['Close']
    # data['Different_MA60'] = (data['MA60'] - data['Close'])/data['Close']
    data = pd.DataFrame(data)
    data.dropna(inplace=True)

    # 4. Labels are the values we want to predict
    diff = np.array(data['dif'])
    price = np.array(data['Close'])

    labels = np.zeros(len(diff))

    for i in range(len(diff)):
        if diff[i] > 0:
            labels[i] = 1

    labels = pd.DataFrame(labels, columns=['labels'])
    # .shift(-1)

    # 5. Data for the features
    features = data[['Close', 'MACD', 'STOCH', 'WILLR', 'RSI']]
    # 'ATR', 'OBV', 'Different_MA20', 'Different_MA60', 'MOM'
    # features = features.set_index('Date')

    return features, labels

# Initial top 30 company data
companylist = ['AXIA', 'CIMB', 'DIAL', 'DSOM', 'HAPS', 'HTHB', 'SUPM', 'TLMM',
               'HLBB', 'HLCB', 'IHHH', 'IOIB', 'KLKK', 'KRIB', 'MBBM', 'NESM', 'MXSC', 'MISC', 'PCGB', 'QRES',
               'PETR', 'PGAS', 'PEPT', 'PMET', 'PUBM', 'RHBC', 'SIME', 'SIPL', 'TENA', 'TPGC', ]

# Collect data for all company into a set/excel 
def collect_whole(companylist):
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    for company in companylist:
        features, labels = get_feature(company, '01/01/2010', '30/04/2019')
        df1 = df1.append(features[:-2])
        df2 = df2.append(labels[:-2])
    df1.to_excel("train_x.xlsx".format(company), index=False)
    df2.to_excel("train_y.xlsx".format(company), index=False)
    return df1, df2

collect_whole(companylist)

# Collect data for each company in separate set/excel
def collect(company):
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    print(company)
    features, labels = get_feature(company, '01/09/2019', '30/12/2020')
    df1 = df1.append(features[:-2])
    df2 = df2.append(labels[:-2])
    df1.to_excel("index_data/{}_x_q4.xlsx".format(company), index=False)
    df2.to_excel("index_data/{}_y_q4.xlsx".format(company), index=False)
    return df1, df2

for company in companylist:
    collect(company)

# Collect closing price of whole company in a list
def get_stocklist_closing_price(companylist, start_date, end_date):

    df_company_closing = pd.DataFrame()
    for company in companylist:
        print(company)
        data = get_stock_data(company, start_date, end_date)
        data.reset_index(inplace=True)
        data = data.replace(0, np.nan)
        data.dropna(inplace=True)
        df_company_closing[company] = data['Close']
    df_company_closing.to_excel(
        "company_closing.xlsx".format(company), index=False)
    return df_company_closing

get_stocklist_closing_price(companylist, '01/01/2010', '30/12/2020')

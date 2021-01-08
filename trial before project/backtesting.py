import numpy as np
import investpy
import talib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from hmmlearn.hmm import GaussianHMM


def get_stock_data(a):
    df = investpy.get_stock_historical_data(stock= a,
                                        country='malaysia',
                                        from_date='01/01/2015',
                                        to_date='30/09/2020')
    return df

def get_feature(a):
    #1. data gathering & processing
    data = get_stock_data(a)
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
    MA60 = talib.MA(data['Close'], timeperiod=60, matype=0)
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
    return features, labels, feature_list, diff, price, MA60

def predict_rf(features, labels, feature_list):
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
 
    return predictions[-1], predictions, test_labels

def predict_svc(features, labels, feature_list):
    #7. train and fit
    # Convert to numpy array
    features = np.array(features)
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features,labels, test_size = 0.25,shuffle=False)
    clf = svm.SVC(C = 10, degree= 4)  # class 
    clf.fit(train_features, train_labels)  # training the svc model'
    predictions = clf.predict(test_features)

    return predictions[-1], predictions, test_labels
 
def predict_adaboost(features, labels, feature_list):
    #7. train and fit
    # Convert to numpy array
    features = np.array(features)
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features,labels, test_size = 0.25,shuffle=False)
    # Instantiate model 
    ada = AdaBoostClassifier()
    # Train the model on training data
    ada = ada.fit(train_features, train_labels)
    # Use the forest's predict method on the test data
    predictions = ada.predict(test_features)
 
    return predictions[-1], predictions, test_labels


def calculate_accuracy(predictions, test_labels):
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
        elif predictions[i] == 0 and test_labels[i] == 1:
            false_negative += 1

    # print(false_negative+true_negative, true_positive+false_positive, len(test_labels))
    accuracy = (true_positive + true_negative)/len(test_labels)
    # print(company ," Accuracy = " , accuracy)
    return accuracy

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
    return SignalRe[-1], IndexRe[-1]

def modifySignal(MA60, price, predictions):
    price = price[-len(predictions):]
    MA60 = MA60[-len(predictions):]
    for i in range(1,len(predictions)):
        if price[i] < MA60[i]:
            predictions[i] = 0
    return predictions

Bluechips = ['MBBM','TPGC','PUBM','TENA','HTHB','PCGB','IHHH','MXSC',
'SIPL','PGAS','NESM','DSOM','HLBB','MISC','CIMB','IOIB',
'PEPT','AXIA','SUPM','KLKK','DIAL','PMET','PETR','KRIB',
'RHBC','HAPS','HLCB','SIME','QRES','TLMM']

Midcaps = ['KLCC','WPHB','GENT','FRAS','GENM','GAMU','AMMB','GENP',
'INAR','MAHB','TCOM','YTLS','SWAY','BMYS','IGRE','HEIN',
'VTRX','BIMB','CBMS','UTPS','BTKW','YINS','STIK','SERB',
'YTLP','FCDY','LOND','IJMS','LOTT','IOIP','MYEG','PREI',
'SUNW','MALA','VSID','GREA','TAKA','ASTR','SIPR','FGVH',
'DRBM','FRKN','MPIM','PMAS','KPJH','MBSS','GASM',
'ALLI','MEGA','GNCH','OTLS','MITE','SETI','UOAD',
'MAGM','AFIN','BATO','UNSM','SETIq','BSTB','UMWS','LEOG',
'ANCR','DBMS','DUOP','AINM','HLIB','SCOG']

Smallcaps = ['IGBB','SKPR','MMCB','COMF','ATAI','SELS','LTKH','MYRS',
'AIRA','SOPS','GDEX','DUFU','SHGM','CARE','GNIC','SAEN',
'PMMY','UMSB','BPOT','TWRK','RUMM','UEME','MAHS','APER',
'BUAB','SEVE','GHLS','JCYI','MALY','OSKH','BERA','GLOA',
'FEHS','WIDA','TAGL','PDNI','IJMP','YNHB','DSON','MATR',
'TSHR','EKOV','SCTH','HAPP','KIML','KSMS','BOUS','CMSM',
'CAMA','PHMA','KREJ','YTLR','KREK','DOVT','UCHI','TAAN',
'TROP','MBMR','BOPL','ECOW','VELE','AEOM','TAES','MATE',
'TGIB','DEHB','UMRS','JHMC','ECOD','HLTG','LBSBq',
'ATLA','BGRO','AJIN','RANH','SAEG','PTMR','KWNF','DAIB',
'IRIB','KEJU','POWE','JFTB','BLAD','MQRE','TMCN','HENY',
'AMWA','HIBI','SEDU','BLDN','CHIU','METR','HSIB',
'LCBH','NTPM','KARE']

# frame bulechips company data
df1 = pd.DataFrame(columns=['companyname'])
df1['companyname'] = Bluechips
# frame midcaps company data
df2 = pd.DataFrame(columns=['companyname'])
df2['companyname'] = Midcaps
# frame smallcaps company data
df3 = pd.DataFrame(columns=['companyname'])
df3['companyname'] = Smallcaps

# calculate the accuracy by difference of closing price within 1-20days
# calculate 30 company accuracy 
def execute(dataframe1,companylist):
    accuracy1 = []
    SignalRe1 = []
    SignalRe2 = []
    IndexRe1 = []
    for a in companylist:        
        features, labels, feature_list, diff, price,  MA60 = get_feature(a)
        trend,predictions, test_labels = predict_adaboost(features,labels,feature_list)
        accuracy1.append(calculate_accuracy(predictions, test_labels))
        SignalRe, IndexRe = backtest(predictions,diff)
        SignalRe1.append(SignalRe)
        IndexRe1.append(IndexRe)
        modifySignal(MA60, price, predictions)
        SignalRe, IndexRe = backtest(predictions,diff)
        SignalRe2.append(SignalRe)
    dataframe1['Accuracy'] = accuracy1
    dataframe1['SignalReturn'] = SignalRe1
    dataframe1['SignalReturn/MA'] = SignalRe2
    dataframe1['IndexReturn'] = IndexRe1


# execute(df1,Bluechips)
# df1.to_excel("adabacktestklci.xlsx") 
# execute(df2,Midcaps)
# df2.to_excel("adabacktestmidcaps.xlsx") 
# execute(df3,Smallcaps)
# df3.to_excel("adabacktestsmallcaps.xlsx") 

features, labels, feature_list, diff, price,  MA60 = get_feature('PMAS')
trend,predictions, test_labels = predict_adaboost(features,labels,feature_list)
print(predictions)
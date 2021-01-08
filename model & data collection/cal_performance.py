from sklearn.metrics import confusion_matrix, precision_score, recall_score, plot_roc_curve, accuracy_score
import numpy as np
import pandas as pd

# import model of random forest
from joblib import dump, load
filename = 'finalized_randomforest_model.joblib'
rf = load(filename)

# calculate cumulative return
def cal_return(predictions, diff):
    period = len(diff) - len(predictions)
    sReturn = diff[period:]

    # strategy cumulative return
    SignalRe = np.cumprod(predictions*(sReturn.shift(-1))+1)-1
    IndexRe = np.cumprod(sReturn+1)-1  # index cummulative return
    SignalRe = np.array(SignalRe)
    IndexRe = np.array(IndexRe)
    return SignalRe[-2], IndexRe[-1]

# strategy from bayes theorem
def strategy_draft(prediction, label):
    label = label.iloc[:, 0]
    action = np.zeros(len(prediction))
    for i in range(len(prediction)):
        if prediction[i] == 1 and label[i] == 0:
            action[i] = 1
        elif (prediction[i] == 0 and label[i] == 0) or (prediction[i] == 0 and label[i] == 1):
            action[i] = 0
        else:
            action[i] = None

    if np.isnan(action[0]):
        action[0] = 0
    print(action[0])
    action = pd.DataFrame(action)
    action = action.fillna(method='ffill')
    action = np.array(action.iloc[:, 0])
    return action

# frame top 30 company data
companylist = ['AXIA', 'CIMB', 'DIAL', 'DSOM', 'HAPS', 'HTHB', 'SUPM', 'TLMM',
               'HLBB', 'HLCB', 'IHHH', 'IOIB', 'KLKK', 'KRIB', 'MBBM', 'NESM', 'MXSC', 'MISC', 'PCGB', 'QRES',
               'PETR', 'PGAS', 'PEPT', 'PMET', 'PUBM', 'RHBC', 'SIME', 'SIPL', 'TENA', 'TPGC', ]
df1 = pd.DataFrame()


for company in companylist:
    print(company)
    test_x = pd.read_excel('index_data/{}_x_q4.xlsx'.format(company))
    test_y = pd.read_excel('index_data/{}_y_q4.xlsx'.format(company))

    # calculate different and remove closing price and date
    diff = test_x['Close'].diff()/test_x['Close'].shift(1)
    test_x1 = test_x.reset_index(drop=True)
    test_x1 = test_x1.drop(['Close'], axis=1)

    # predict and calculate accuracy, prediction and return
    y_pred = rf.predict(test_x1)
    y_pred = strategy_draft(y_pred, test_y)
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred)
    SignalRe, IndexRe = cal_return(y_pred, diff)
    df1 = df1.append([[accuracy, precision, SignalRe, IndexRe]])

    # save company's prediction
    # test_x['prediction'] = y_pred
    # test_x.to_excel("{}_prediction.xlsx".format(company), index = False)


# output performance in excel
df1.columns = ['Accurracy', 'Precision', 'Signal_return', 'Index_return']
df1['companyname'] = companylist
df1 = df1.set_index('companyname')
df1.to_excel("index_data/strategy_performance_q4.xlsx")

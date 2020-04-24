import datetime
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from pandas.plotting import register_matplotlib_converters
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import scale
from pandas.plotting import register_matplotlib_converters


#1. data gathering & processing
data = pd.read_csv('data/SIME.csv')
data = data.replace(0, np.nan)
data.dropna(inplace=True)
data['Date'] = pd.to_datetime(data.Date, format= '%Y-%m-%d')

#2. exponential smoothing
S = X = np.array(data['Close'])
alpha = 0.8
for i in range(1,len(S)):
    S[i] = alpha*X[i] + (1-alpha)*S[i-1]
data['Close'] = S

#3. feature extraction
data['V/MA']=data['Volume']/data['Volume'].rolling(15).mean()
data=data.set_index('Date')
data=data['2016':'2020']

O1=0.4*((data['Close']-data['Open'])/(data['High']-data['Low'])).values
O1.shape=(len(O1),1)
O1=np.log((1+O1)/(1-O1))

O2=data['V/MA'].values/(data['High']/data['Low']).values
O2.shape=(len(O2),1)
O2=0.2*np.log(O2)
O2=np.log((1+O2)/(1-O2))

#Ob1=np.hstack((O1,O2))
Ob1=O1 #observe data of hmm1
Ob2=O2 #observe data of hmm2

#5. split data
Return=(data['Close']/data['Open']-1).values #daily return
len1=len(data[:'2019']) 
len2=len(data['2020':]) 
Signal=np.zeros(len2)
np.random.seed(1)
N_state1=3 #first hmm 3 state
N_state2=4 #second hmm 4 state

#4. Labels are the values we want to predict
data['dif'] = data['Close'].diff(-1)
diff = np.array(data['dif'])
labels = np.zeros(len(diff))
for i in range(len(diff)):
    if diff[i] > 0:
        labels[i] = 1
labels = labels[len1:]

#4. fit and train model
for i in range(len2):
    #rolling training every month
    if data.index[len1+i-1].month!=data.index[len1+i].month: 
        remodel1=GaussianHMM(n_components=N_state1)
        remodel1.fit(Ob1[:len1+i])
        remodel2=GaussianHMM(n_components=N_state2)
        remodel2.fit(Ob2[:len1+i])
    s_pre1=remodel1.predict(Ob1[:len1+i]) #prediction on historical data
    s_pre2=remodel2.predict(Ob2[:len1+i])
    Re=Return[:len1+i] #historical return
    #state mean return
    Expect=np.array([np.mean(Re[(s_pre1==j)*(s_pre2==k)]) \
                     for j in range(N_state1) for k in range(N_state2)])
    #probability of event happening
    Pro=np.array([remodel1.transmat_[s_pre1[-1],j]*remodel2.transmat_[s_pre2[-1],k]\
                  for j in range(N_state1) for k in range(N_state2)])
    preReturn=Pro.dot(Expect) #prediction of tommorow return
    if preReturn>0.:
        Signal[i]=1

sReturn=(data['Close']/data['Close'].shift(1))[len1:]-1 #backtesting period return
#cost of each trade
Cost=pd.Series(np.zeros(len(Signal)),index=sReturn.index) 
for i in range(1,len(Signal)):
    if Signal[i-1]!=Signal[i]:
        Cost.values[i]=0.0006
SignalRe=np.cumprod(Signal*(sReturn-Cost)+1)-1 #strategy cumulative return
IndexRe=np.cumprod(sReturn+1)-1 #index cummulative return
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(IndexRe,label='Index')
plt.plot(SignalRe,label='Signal')
plt.plot(SignalRe-IndexRe,label='excess')
plt.legend()
plt.show()

print("Tommorow's trend:", end =" ")
probability = 0
if Signal[-1] == 1:
    print("Uptrend")
    print("The probability of trend:" , end =" ")
    for i in range(len(Pro)):
        if Expect[i] > 0:
            probability += Pro[i]
    print(probability)
else:
    print("Downtrend")
    print("The probability of trend:" , end =" ")
    for i in range(len(Pro)):
        if Expect[i] < 0:
            probability += Pro[i]
    print(probability)
print("Expected return:", end =" ")
print(Expect[-1])

#6. Get accuracy, precision, recall and specificity
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
for i in range(len(Signal)):
    if Signal[i] == 1 and labels[i] == 1:
        true_positive += 1
    elif Signal[i] == 1 and labels[i] == 0:
        false_positive += 1
    elif Signal[i] == 0 and labels[i] == 0:
        true_negative += 1
    elif Signal[i] == 0 and labels[i] == 0:
        false_negative += 1

accuracy = (true_positive + true_negative)/(len(labels))
precision = true_positive/(true_positive + false_positive)
recall = true_positive/(true_positive + false_negative)
specificity = true_negative/(true_negative + false_positive)

print(false_negative)
print("Accuracy = " , accuracy)
print("Precision = " , precision)
print("Recall = " , recall)
print("Specificity = " , specificity)

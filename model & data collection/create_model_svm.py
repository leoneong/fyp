import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, plot_roc_curve, accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint

train_x = pd.read_excel('f_data/train_x.xlsx')
train_y = pd.read_excel('f_data/train_y.xlsx')
test_x = pd.read_excel('f_data/test_x.xlsx')
test_y = pd.read_excel('f_data/test_y.xlsx')

# support vector machine fit
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf = clf.fit(train_x, train_y.values.ravel())

from joblib import dump, load
filename = 'finalized_svm_model.joblib'
# save the model to disk
dump(clf, filename)
# load file
clf = load(filename)

# plot roc
ax = plt.gca()
cl_disp = plot_roc_curve(clf, test_x, test_y, ax=ax, alpha=0.8)
plt.show()


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, plot_roc_curve, accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint

train_x = pd.read_excel('f_data/train_x.xlsx')
train_y = pd.read_excel('f_data/train_y.xlsx')
test_x = pd.read_excel('f_data/test_x.xlsx')
test_y = pd.read_excel('f_data/test_y.xlsx')

# normal random forest
rf = RandomForestClassifier(n_estimators = 100, oob_score = True, n_jobs = -1, random_state=0, max_features = "auto", min_samples_leaf = 50)
rf = rf.fit(train_x, train_y.values.ravel())

# print random forest importance
print(rf.feature_importances_)

from joblib import dump, load
filename = 'finalized_randomforest_model.joblib'
filename1 = 'finalized_svm_model.joblib'
# save the model to disk
dump(rf, filename)
# load file
rf = load(filename)
clf = load(filename1)


# grid search for random forest
from sklearn.model_selection import RandomizedSearchCV
param_grid = {"max_depth": sp_randint(1, 8),                     
              "max_features": sp_randint(1, 4),          
              "min_samples_split": sp_randint(2, 4),    
              "bootstrap": [True, False],   
              'n_estimators':[1, 2, 4, 8, 16, 32, 64, 100, 200],            
              "criterion": ["gini", "entropy"]}        

random_search = RandomizedSearchCV(rf, param_grid, cv = 4)
random_search.fit(train_x,train_y.values.ravel())
y_random_pred = random_search.predict(test_x)
con_m1 = confusion_matrix(test_y, y_random_pred)
precision1 = precision_score(test_y, y_random_pred)

# plot roc
ax = plt.gca()
rf_disp = plot_roc_curve(rf, test_x, test_y, ax=ax, alpha=0.8) 
clf_disp = plot_roc_curve(clf, test_x, test_y, ax=ax, alpha=0.8)
plt.show()

# cross validation get accuracy
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=rf, X=train_x, y=train_y.values.ravel(), cv = 10)
print('rf:')
print(all_accuracies)

all_accuracies_svm = cross_val_score(estimator=clf, X=train_x, y=train_y.values.ravel(), cv = 10)
print('clf:')
print(all_accuracies_svm)


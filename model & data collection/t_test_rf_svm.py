import numpy as np
from scipy import stats

# Take in accuracy of model by cross validation of 10-fold
rf = np.array([0.81163624, 0.81965535, 0.8104419, 0.81194539, 0.79948805, 0.81365188,
 0.78703072, 0.79419795, 0.80904437, 0.79283276])

clf = np.array([0.80702952, 0.82238526, 0.80788261, 0.8116041,  0.8,    0.81075085,
 0.77133106, 0.78634812, 0.81877133, 0.79180887])

#Calculate the variance to get the standard deviation
#For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
var_a = rf.var(ddof=1)
var_b = clf.var(ddof=1)

#std deviation
s = np.sqrt((var_a + var_b)/2)

# Calculate the t-statistics
t = (rf.mean() - clf.mean())/(s*np.sqrt(2/10))
print("t_test = " + str(t))

# get t & p value of t-test
t2, p2 = stats.ttest_ind(rf,clf)
print("t_critical = " + str(t2))
print("p_value = " + str(p2))
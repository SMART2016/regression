# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using backward elimination
# the library for creating model do not handle the constant b0 so need to add to the dataset.
import statsmodels.formula.api as sm
#np.append([[1,2],[3,4]],[[1],[10]], axis=1)
#O/p: array([[ 1,  2,  1],
#       [ 3,  4, 10]])
#addding constant b0 to the X data set
X = np.append(arr = np.ones((50,1)).astype(int),values= X,axis = 1)
# Step 1: 
X_OPT = X[:,[0,1,2,3,4,5]]
# Step 3: calculating the stats for the sample
regressor_OLS = sm.OLS(endog=y ,exog= X_OPT).fit()

#Step 4: Iterative step of removing predictors or independent variables one by one based on there P-Value > 0.05,
#until the remaining variables P-Value is less than 0.05
# also if the adjusted Rsquare before removal of the variable is greater than the adj-R^2 after removal that means ,
# we should not remove the variable from the predictor set.
pVals = regressor_OLS.pvalues
sigLevel = 0.05
while pVals[np.argmax(pVals)] > sigLevel:
     X_OPT = np.delete(X_OPT, np.argmax(pVals), axis = 1)
     print("pval of dim removed: " + str(np.argmax(pVals)))
     print(str(X_OPT.shape[1]) + " dimensions remaining...")
     regressor_OLS = sm.OLS(endog = y, exog = X_OPT).fit()
     pVals = regressor_OLS.pvalues

regressor_OLS.summary()


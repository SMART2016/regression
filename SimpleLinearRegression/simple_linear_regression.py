# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set(context="notebook", palette="deep",style = 'darkgrid' ,font_scale = 1.5, color_codes=True)

#1) Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#Information about the datatype of the fields of the dataset
dataset.info()


#Assumption 1: Linearity check: The output graph shows a linear pattern between Salary and yearsOfExperience
p = sns.pairplot(dataset, x_vars=['YearsExperience'], y_vars='Salary', size=7, aspect=0.7)

#Get Dependent(Y) and independent(X) variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#2) Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#3) Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#4) Fitting Simple Linear Regression to the Training set to get the best fit constant b0 and coefficient b1
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#5) Predicting the Test set results
y_pred = regressor.predict(X_test)

Y_PRED = regressor.predict(X)

#Assumption 2: Mean of residual should be 0 or close to 0 : its not close to 0 ,
#so Linear regression should not be applied to the dataset
residual = y -  Y_PRED
mean_residual = np.mean(residual)
print("Mean of Residuals {}".format(mean_residual))


#Assumption 3: Residuals should be normally distributed
import scipy.stats as stats
import statsmodels.api as sm
fig = sm.qqplot(residual, stats.t, distargs=(4,))
plt.show()

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
#It doesnot matter if we use test set or training set because the line will be same
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
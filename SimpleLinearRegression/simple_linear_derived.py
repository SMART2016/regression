# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean


#1) Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#Information about the datatype of the fields of the dataset
dataset.info()

#2) Get Dependent(Y) and independent(X) variables
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

#3) get the slope and Y-intercept
b1,b0 = best_fit_slope_and_intercept(X,y)

#4) Get predicted Y values or the regression line which will predict data
regression_line = [(b1*X)+b0 for x in X]

#5) R-squared for the regression line if its closer to 1 that means error is low and line is a good fit
r_squared = coeeficient_of_determination(y,regression_line[0])
print(r_squared)

#6) predicted new Y
y1 = predict(9,b1,b0)


#7) Plot the graph of actual data and predicted best fit line
plt.scatter(X,y)
plt.scatter(9,y1,color = 'r')
plt.plot(X,regression_line[0])
plt.show()


#Functions --------------------------------------------------

#predict function
def predict(x,m,c):
    return m * x + c

#function to calculate slope and Y-intercept from dataset
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)
    
    return m, b

#function to return suared error
def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) ** 2)

#coefficient of determination r^2
def coeeficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    sqr_err_reg_line = squared_error(ys_orig,ys_line)
    sqr_err_mean_line = squared_error(ys_orig,y_mean_line)
    
    return 1 - (sqr_err_reg_line/sqr_err_mean_line)

# Data Preprocessing Template

# Importing the libraries
#numpy: used for mathematical functions
import numpy as np
#Used for plotting charts
import matplotlib.pyplot as plt
#Used for importing and managing datasets
import pandas as pd

#1) Importing the dataset ---------------------
dataset = pd.read_csv('Data.csv')

#2) Defining Dependent (Target) and Independent (Feature) Variables -----------------------
#The first : in iloc on the left of comma means all rows of the dataset
#The next :-1 on the right of comma means all columns exept the last one
X = dataset.iloc[:, :-1].values
#All rows with just the last column total 4 columns indexed from 0 to 3
y = dataset.iloc[:, 3].values


#3) [Optional] Replace missing data , a good way is to replace the missing data with the mean for that column.
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])"""
#X[:,1:3] = imputer.transform(X[:,1:3])


#4) [Optional] Encode Categorical variables Country and Purchased from X and y variables as they are non-numeric
#LabelEncoder: encodes the columns with simple numeric values , in which case the ML algorithm can deduce relation between them.
#OneHotEncoder: encodes the columns with dummy values , which prevents the above problem with LabelEncoder.
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
labelEncoder_Y = LabelEncoder()
y = labelEncoder_Y.fit_transform(y)
#Dummy variabkle encoding
oneHotEncoder_X = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder_X.fit_transform(X).toarray()"""

#5) Splitting the dataset into the Training set and Test set
#Test set is chosen as a smaller percent of the whole data close to 20 to 30% of the whole dataset.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#6) Feature Scaling: scaled between -1 to 3, not required with all algorithms
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)"""
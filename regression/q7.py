# Estime a capacidade de generalização de um regressor linear, 
# Y=f(X1, X2, ..., X5), sem usar regularização, via validação cruzada 
# Leave-One-Out.


# import libraries
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut


# get data from csv file
df = pd.read_csv('regressao.csv')


# convert data to array format
data = df.as_matrix()


# select input and output from data
X = data[:,0:-1]
Y = data[:,-1]




model = LinearRegression()

model.fit(X, Y)

Ypred = model.predict(X)


print(Ypred)

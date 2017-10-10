# Estimate a generalization measure from the linear regression model  
# Y=f(X1, X2, ..., X5) with no regularization, and using 
# leave-one-out cross-validation.



# import libraries
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error


# get data from csv file
df = pd.read_csv('regressao.csv')


# convert data to array format
data = df.as_matrix()


# select input and output from data
X = data[:,0:-1]
Y = data[:,-1]



model = LinearRegression()
# model.fit(X, Y)
# Ypred = model.predict(X)

# print(Ypred)



# create a leave-one-out object
loo = LeaveOneOut()

# create a list to store all rmse values
rmse = []


# Leave-one-out cross-validation loop 
for train_index, test_index in loo.split(X):
	X_train, Y_train = X[train_index], Y[train_index]	
	X_test, Y_test = X[test_index], Y[test_index]
	model.fit(X_train, Y_train)
	Ypred = model.predict(X_test)
	rmse.append( np.sqrt(((Ypred-Y_test)**2).mean()) )
	# rmse.append(np.sqrt(mean_squared_error(Y_test, Ypred)))


# compute mean cross-validation metric
rmse = np.mean(rmse)
print('\nLeave-one-out cross-val rmse: %.3f \n' % (rmse))

# Utilizando um regressor (não paramétrico) k-NN, com k=5 vizinhos e 
# baseado em distâncias euclidianas, obter Y para a tupla 
# [245,4,9700,4600,1835,?].



# import libraries
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import LeaveOneOut




# get datadata from csv file
df = pd.read_csv('regressao.csv')


# convert data to array format
data = df.as_matrix()


# select input and output from data
X = data[:,0:-1]
Y = data[:,-1]




# create a KNN regression model
neigh = KNeighborsRegressor(n_neighbors=5)



# train the model with all data
neigh.fit(X, Y)


# input sample to predict
xpred = np.array([245,4,9700,4600,1835])
ypred = neigh.predict(np.reshape(xpred, (1, -1) ))


print('\nInput sample:\n', xpred)
print('\nPredicted value by regression tree: %.3f' % (ypred))






#####################################################################
# uncomment down here to check the one-leave-out cross validation analysis


# Ypred = neigh.predict(X)
# print('Training rmse: %.3f' % (np.sqrt(((Y-Ypred)**2).mean())))


# # create a leave-one-out object
# loo = LeaveOneOut()

# # create a list to store all rmse values
# rmse = []


# # Leave-one-out cross-validation loop 
# for train_index, test_index in loo.split(X):
# 	X_train, Y_train = X[train_index], Y[train_index]	
# 	X_test, Y_test = X[test_index], Y[test_index]
# 	neigh.fit(X_train, Y_train)
# 	Ypred = neigh.predict(X_test)
# 	rmse.append( np.sqrt(((Ypred-Y_test)**2).mean()) )
# 	# rmse.append(np.sqrt(mean_squared_error(Y_test, Ypred)))


# # compute mean cross-validation metric
# rmse = np.mean(rmse)
# print('\nLeave-one-out cross-val rmse: %.3f \n' % (rmse))


# Classify the tuple [ensolarado, quente, normal, verdadeiro] using 
# Naïve Bayes classifier.



# import libraries
import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB




# get datadata from csv file
df = pd.read_csv('cinco.csv')


# convert data to array format
data = df.as_matrix()


# select input and output (nominal data)
Xnom = data[:,0:-1]
Ynom = data[:,-1]



# create array to store numerical-converted data
X = -1*np.ones(Xnom.shape)




# convert categorical input data to numerical type
for i in range(Xnom.shape[1]):
	classes = list(np.unique(Xnom[:,i]))
	# ls_index = np.arange(0,len(classes))
	X[:,i] = list(map(lambda x: classes.index(x), list(Xnom[:,i])))




# convert categorical output data to numerical type
Y = list(map(lambda y: ['Nao', 'Sim'].index(y), list(Ynom)))





# create a gaussian Naive Bayes model
gnb = GaussianNB()



# train the model with all data
gnb.fit(X, Y)




# tuple value to predict using the trained model
xpred = ['Ensolarado', 'Quente', 'Normal', 'Verdadeiro']
print('\nInput sample: \n', xpred)



# convert categorical tuple to numerical array
xpred_nom = []
for i in range(Xnom.shape[1]):
	classes = list(np.unique(Xnom[:,i]))
	xpred_nom.append(classes.index(xpred[i]))
xpred_nom = np.array(xpred_nom)




# predict the output for given input tuple
ypred_nom = gnb.predict(np.reshape(xpred_nom, (1,-1)))
print('\nPredicted value by Naive Bayes classifier: ',
	['Nao', 'Sim'][int(ypred_nom)])



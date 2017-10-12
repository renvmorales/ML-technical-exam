# Use a decision tree classifier, with no pruning, and entropy gain 
# criterion to classify the tuple [Sim,Sim,Cheio,Francês].



# import libraries
import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier




# get datadata from csv file
df = pd.read_csv('seis.csv', sep=';')


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




# create the model object
treeClass = DecisionTreeClassifier(criterion='entropy')




# train the model with all data
treeClass.fit(X, Y)





# tuple value to predict using the trained model
xpred = ['Sim','Sim','Cheio','Frances']
print('\nInput sample: \n', xpred)





# convert categorical tuple to numerical array
xpred_nom = []
for i in range(Xnom.shape[1]):
	classes = list(np.unique(Xnom[:,i]))
	xpred_nom.append(classes.index(xpred[i]))
xpred_nom = np.array(xpred_nom)






# predict the output for given input tuple
ypred_nom = treeClass.predict([xpred_nom])
print('\nPredicted value by a Decision Tree classifier: ',
	['Nao', 'Sim'][int(ypred_nom)])

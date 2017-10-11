# Obter a árvore de decisão, sem nenhuma poda e selecionando-se os 
# atributos pelo critério do ganho da informação, para classificar a 
# tupla [Sim,Sim,Cheio,Francês].



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




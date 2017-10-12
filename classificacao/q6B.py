# Deseja-se classificar um novo exemplo xt=[Sim,Sim,Alguns,Italiano,?]. 
# Considere que todos os seus atributos sejam nominais. Pede-se 
# classificar xt de acordo com o método dos vizinhos mais próximos 
# (k-NN, com k=3). Utilize a medida de dissimilaridade baseada no 
# coeficiente de casamento simples para um espaço p-dimensional



# import libraries
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier




# get datadata from csv file
df = pd.read_csv('seis.csv', sep=';')


# convert data to array format
data = df.as_matrix()


# select input and output (nominal data)
Xnom = data[:,0:-1]
Ynom = data[:,-1]



# create array to store numerical-converted data
X = -1*np.ones(Xnom.shape)





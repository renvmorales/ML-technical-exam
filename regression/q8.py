# Construa uma árvore de regressão para estimar o valor de Y na tupla 
# [245,4,9700,4600,1835,?]. Use todos os dados disponíveis e faça com 
# que a altura máxima da árvore seja igual a 3.


# import libraries
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error



# get datadata from csv file
df = pd.read_csv('regressao.csv')


# convert data to array format
data = df.as_matrix()


# select input and output from data
X = data[:,0:-1]
Y = data[:,-1]

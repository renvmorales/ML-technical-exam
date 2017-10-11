# Classificar a tupla [ensolarado, quente, normal, verdadeiro] pelo 
# classificador bayesiano simples (Naive Bayes).




# import libraries
import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import LeaveOneOut





# get datadata from csv file
df = pd.read_csv('cinco.csv')


# convert data to array format
data = df.as_matrix()


# select input and output from data
X = data[:,0:-1]
Y = data[:,-1]








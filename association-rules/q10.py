# Considerando a tabela acima, contendo 10 transações, para as quais 
# S (sim) e N (não) significam respectivamente a ocorrência ou não de um 
# determinado item numa transação, obter o suporte e a confiança da regra
# “Se {manteiga, pão} então {café}”.



# import libraries
import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules




# get datadata from csv file
df = pd.read_csv('dez.csv')




# create a encoder function for each transaction
def encode_units(x):
    if x == 'N':
        return 0
    if x == 'S':
        return 1


# apply the encoder to the dataframe
df_onehot = df.applymap(encode_units)
print('Original Transaction table:\n')
print(df_onehot.head())




# list all support values in descending order
#    OBS: support here is computed with respect to a single, or a set
#    items. However, to find a support rule just assemble antescedant and
# 	 consequent items as a single item set.
fq_itemsets = apriori(df_onehot, min_support=0.3, use_colnames=True)
print('\nFrequent item sets:\n', 
	fq_itemsets.sort_values(by='support', ascending=False).head(10))




# list confidence and lift metrics with respect to the association rules
rules = association_rules(fq_itemsets, metric="confidence", min_threshold=0.3)
print('\nAssociation rules:\n',
	rules.sort_values(by='lift', ascending=False).head(10))

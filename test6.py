# data cleanse for tableau 

import pandas as pd 
import numpy as np 

df = pd.read_csv('dashride_dataframe.csv', sep = ',', index_col = 1)
df['order'] = 'start'
order = df['order'].copy()

for i in range(30000, 60000):
	order.iloc[i] = 'end'

df['order'] = order
# print df['order'].describe()

df.to_csv('dashride_tableau.csv', sep = ',')

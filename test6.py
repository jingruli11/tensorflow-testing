<<<<<<< HEAD
import pandas as pd 
import numpy as np 

df = pd.read_csv('turnstile_170624.txt', sep = ',')
df = df.append(pd.read_csv('turnstile_170617.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170610.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170603.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170527.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170520.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170506.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170401.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170408.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170415.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170422.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170429.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170325.txt', sep = ','))

print(len(df))

=======
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
>>>>>>> origin/master

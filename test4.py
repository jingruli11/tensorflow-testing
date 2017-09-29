# dashride project dataset preprocessing

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# import tensorflow as tf 


# read csv file into pandas dataframe
df = pd.read_csv('dashride_columbia_dataset.csv')

# split the string in starting and ending location to create coordinates variables
df['start_lt'] = df['startLoc'].apply(lambda x: x.split(';')[0])
df['start_lg'] = df['startLoc'].apply(lambda x: x.split(';')[1])
df['end_lt'] = df['endLoc'].apply(lambda x: x.split(';')[0])
df['end_lg'] = df['endLoc'].apply(lambda x: x.split(';')[1])

# delete the startloc and endloc variables
columns_to_keep = ['reservationNumber', 'from', 'passengers', 'rider', 
					'serviceLevel', 'scheduledTime', 'start_lt','start_lg',
					'end_lt','end_lg']
df = df[columns_to_keep].dropna()
df['serviceLevel'].replace(to_replace = '571f94066095b2634daa3f7a', value = 'Any vehicle', inplace = True)
df['serviceLevel'].replace(to_replace = '5720cacf9b36887aef7fbaa6', value = 'Wheelchair-accessible', inplace = True)
df['serviceLevel'].replace(to_replace = '5728a0b84b4be25e7aec2a8c', value = 'Any vehicle', inplace = True)
df['serviceLevel'].replace(to_replace = '572e01e12db0c8729f9859af', value = 'Van', inplace = True)
df['serviceLevel'].replace(to_replace = '58c7f9f39dc296a494091494', value = 'Any vehicle', inplace = True)

# filter the dataframe to only include service level = any vehicle
df = df[df['serviceLevel'] == 'Any vehicle']

# check the result of filter
# print df['serviceLevel'].describe()

# output the dataframe to csv file
df.to_csv('dataframe.csv', sep = ',', index_label = 'Index')
#print df.head()
#print df['serviceLevel'].value_counts()

'''
# check the basic stat of each variable
for i in columns_to_keep:
	print i + ':\n'
	print df[i].describe()
'''	




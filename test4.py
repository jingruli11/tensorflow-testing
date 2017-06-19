# dashride project dataset preprocessing

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
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
df['serviceLevel'].replace(to_replace = '571f94066095b2634daa3f7a', value = 'standard', inplace = True)
df['serviceLevel'].replace(to_replace = '5720cacf9b36887aef7fbaa6', value = 'SUV/van', inplace = True)


df.to_csv('dataframe.csv', sep = ',')
#print df.head()
#print df['serviceLevel'].value_counts()

'''
# check the basic stat of each variable
for i in columns_to_keep:
	print i + ':\n'
	print df[i].describe()
'''	




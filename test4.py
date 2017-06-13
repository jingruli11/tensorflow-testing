# dashride project dataset preprocessing

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn


# read csv file into pandas dataframe
df = pd.read_csv('dashride_columbia_dataset.csv')

# split the string in starting and ending location to create coordinates variables
df['start_lat'] = df['startLoc'].apply(lambda x: x.split(';')[0])
df['start_long'] = df['startLoc'].apply(lambda x: x.split(';')[1])
df['end_lat'] = df['endLoc'].apply(lambda x: x.split(';')[0])
df['end_long'] = df['endLoc'].apply(lambda x: x.split(';')[1])

# delete the startloc and endloc variables
columns_to_keep = ['reservationNumber', 'from', 'to', 'passengers', 'rider', 
					'serviceLevel', 'scheduledTime', 'start_lat','start_long',
					'end_lat','end_long']
df = df[columns_to_keep]

print df.head()


# check the basic stat of each variable
for i in columns_to_keep:
	print i + ':\n'
	print df[i].describe()
	df[i].plot(kind = 'hist')
	plt.show()
	




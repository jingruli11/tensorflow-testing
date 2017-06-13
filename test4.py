# dashride project dataset preprocessing

import pandas as pd 
import numpy as np 

df = pd.read_csv('dashride_columbia_dataset.csv')
df['start_lat'] = df['startLoc'].apply(lambda x: x.split(';')[0])
df['start_long'] = df['startLoc'].apply(lambda x: x.split(';')[1])
df['end_lat'] = df['endLoc'].apply(lambda x: x.split(';')[0])
df['end_long'] = df['endLoc'].apply(lambda x: x.split(';')[1])

print df.head()



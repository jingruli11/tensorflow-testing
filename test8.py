import pandas as pd 
import numpy as np 
import googlemaps
from googlemaps import Client as GoogleMaps 

gmaps = GoogleMaps('AIzaSyAWfgiQDkcbA6pWQQa4Z_CD3DCzGRS7GQc')

df = pd.read_csv('dataframe.csv', sep = ',', index_col = 0)

# print df.head()

geo_cols = ['start_lt', 'start_lg', 'end_lt', 'end_lg']

df_geo = df[geo_cols]

# print df_geo.head()

# use for loop to generate end location address
overall = []

import time
start_time = time.time()

n = int(np.floor(len(df_geo)/100))
temp = []

print len(df)
'''
for j in range(0,100):
	dest = gmaps.reverse_geocode((df_geo.iloc[j,2],df_geo.iloc[j,3]))
	temp.append(dest[0]['formatted_address'])

print len(temp)
print "--- %s seconds ---" % (time.time() - start_time)



dest = gmaps.reverse_geocode((df_geo.iloc[1,2],df_geo.iloc[1,3]))
temp.append(dest[0]['formatted_address'])

print temp
print "--- %s seconds ---" % (time.time() - start_time)
# df['to'] = df.apply(lambda x: gmaps.reverse_geocode((x['end_lt'],x['end_lg'])), axis = 1)[0]['formatted_address']

#dest = gmaps.reverse_geocode((df_geo.iloc[1,2],df_geo.iloc[1,3]))
#print dest[0]['formatted_address']
# destination.to_csv('dest_testing.csv', sep = ',')
'''





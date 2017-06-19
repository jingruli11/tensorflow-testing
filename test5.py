# Exploratory Data Analysis on CLL project


import pandas as pd 
import numpy as np 
import pyarrow.parquet as pq 
import matplotlib.pyplot as plt 

'''
# read parquet file into dataframe
# one parquet file has more than 100,000 rows, read 8 and combine them together 
# so that the total file has over 1 mill entires
df0 = pq.read_table('/Users/michaelli/Downloads/part-00000-38d7fbb2-5df4-41a0-9c67-57172fa222a2-c000.gz.parquet').to_pandas()
df1 = pq.read_table('/Users/michaelli/Downloads/part-00001-38d7fbb2-5df4-41a0-9c67-57172fa222a2-c000.gz.parquet').to_pandas()
df2 = pq.read_table('/Users/michaelli/Downloads/part-00002-38d7fbb2-5df4-41a0-9c67-57172fa222a2-c000.gz.parquet').to_pandas()
df3 = pq.read_table('/Users/michaelli/Downloads/part-00003-38d7fbb2-5df4-41a0-9c67-57172fa222a2-c000.gz.parquet').to_pandas()
df4 = pq.read_table('/Users/michaelli/Downloads/part-00004-38d7fbb2-5df4-41a0-9c67-57172fa222a2-c000.gz.parquet').to_pandas()
df5 = pq.read_table('/Users/michaelli/Downloads/part-00005-38d7fbb2-5df4-41a0-9c67-57172fa222a2-c000.gz.parquet').to_pandas()
df6 = pq.read_table('/Users/michaelli/Downloads/part-00006-38d7fbb2-5df4-41a0-9c67-57172fa222a2-c000.gz.parquet').to_pandas()
df7 = pq.read_table('/Users/michaelli/Downloads/part-00007-38d7fbb2-5df4-41a0-9c67-57172fa222a2-c000.gz.parquet').to_pandas()

df = df0.append(df1).append(df2).append(df3).append(df4).append(df5).append(df6).append(df7)
print 'Shape:\n'
print df.shape

df.to_csv('cll_df.csv', sep = ',')

df.dropna(inplace = True)
# get details of each column
cols = list(df.columns.values)
for i in cols:
	print df[i].describe()
'''

# read csv file into dataframe
df = pd.read_csv('cll_df_small.csv', sep = ',', index_col = 1)

# create a smaller sample for easy viewing in Excel
# df.iloc[0:100,:].to_csv('cll_df_small.csv', sep = ',')

# create a column of test performed year
df['test_performed_year'] = df['estimated_patient_test_performed_datetime'].apply(lambda x: x.split(' ')[0])
df['test_performed_year'] = df['test_performed_year'].apply(lambda x: x.split('-')[0])
# print df['test_performed_year'].describe()

# select patients diagnosed with CLL
df.loc[((('ICD9/204.1x' | 'ICD10/C91.10' | 'ICD10/C91.11' | 'ICD10/C91.12') in df['diagnosis_codes'] ) 
		| ()) & (df['test_performed_year'] - df['patient_date_of_birth'] > 18)]













import pandas as pd 
import numpy as np 

df = pd.read_table('Medivo_test.txt', sep = '|')

df.to_csv('medivo_test.csv', sep = ',')
